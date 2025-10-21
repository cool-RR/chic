#!/usr/bin/env python
'''
GRPO Demo - Training Gemma3-1b-it on GSM8K math reasoning benchmark
Adapted from Tunix GRPO demo notebook
'''

import functools
import gc
import os
import re
import csv
import shutil
import tempfile
import pathlib
from contextlib import contextmanager
import textwrap

import click
from flax import nnx
import grain
import humanize
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
import qwix
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params as params_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

from chic.trekking import Trek


# ANSI color codes for terminal formatting
class Color:
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'


@contextmanager
def create_temp_folder(prefix=tempfile.template, suffix=''):
    '''
    Context manager that creates a temporary folder and deletes it after usage.

    After the suite finishes, the temporary folder and all its files and
    subfolders will be deleted.
    '''
    temp_folder = pathlib.Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))
    try:
        yield temp_folder
    finally:
        shutil.rmtree(str(temp_folder), ignore_errors=True)


# ============================================================================
# Prompt Templates and Constants
# ============================================================================

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

# Simplified prompt - less demanding for small models
SYSTEM_PROMPT = ('Solve this math problem step by step. At the end, write "The answer is: " '
                 'followed by just the number.')

TEMPLATE = textwrap.dedent('''\
    <start_of_turn>user
    {system_prompt}

    Problem: {question}<end_of_turn>
    <start_of_turn>model
    Let me solve this step by step:
''')


# ============================================================================
# Utility Functions
# ============================================================================

def show_hbm_usage():
    '''Displays memory usage per device.'''
    fmt_size = functools.partial(humanize.naturalsize, binary=True)
    for d in jax.local_devices():
        stats = d.memory_stats()
        used = stats["bytes_in_use"]
        limit = stats["bytes_limit"]
        print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def _as_text(v):
    return v if isinstance(v, str) else v.decode("utf-8")


def download_kaggle_dataset(target_dir="./data/gsm8k"):
    os.makedirs(target_dir, exist_ok=True)
    src = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
    src = pathlib.Path(src)
    dst = pathlib.Path(target_dir)
    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
        print(f"  Copied {csv_file.name} → {dst/csv_file.name}")
    return target_dir


def get_dataset(data_dir, split="train", source="tfds"):
    '''Load and prepare dataset.'''
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if source == "tfds":
        import tensorflow_datasets.text.gsm8k
        data = tfds.data_source(
            "gsm8k",
            split=split,
            data_dir=data_dir,
            builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
            download=True,
        )
    elif source == "kaggle":
        kaggle_dir = download_kaggle_dataset(data_dir)
        file_name = "main_" + split + ".csv"
        csv_path = os.path.join(kaggle_dir, file_name)
        data = []
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append({
                    "question": row["question"],
                    "answer": row["answer"],
                })
    else:
        raise ValueError(f"Unknown source: {source}")

    dataset = (
        grain.MapDataset.source(data)
        .shuffle(seed=42)
        .map(
            lambda x: {
                "prompts": TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=_as_text(x["question"]),
                ),
                "question": _as_text(x["question"]),
                "answer": extract_hash_answer(_as_text(x["answer"])),
            }
        )
    )
    return dataset


# ============================================================================
# Model Loading Functions
# ============================================================================

def get_gemma_ref_model(ckpt_path, mesh_config):
    '''Load Gemma3 model from Orbax checkpoint.'''
    mesh = jax.make_mesh(*mesh_config)
    model_config = gemma_lib.ModelConfig.gemma3_1b()

    # Load from Orbax checkpoint (Kaggle format)
    gemma = params_lib.create_model_from_checkpoint(
        ckpt_path, model_config, mesh
    )

    return gemma, mesh, model_config


def get_lora_model(base_model, mesh, lora_rank, lora_alpha):
    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
            ".*attn_vec_einsum"
        ),
        rank=lora_rank,
        alpha=lora_alpha,
    )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **model_input
    )

    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model


# ============================================================================
# Reward Functions
# ============================================================================

# RegEx for simpler format matching - looks for "The answer is: NUMBER"
match_format = re.compile(
    r"The answer is:\s*(-?[\d,\.]+)",
    flags=re.MULTILINE | re.IGNORECASE,
)

match_numbers = re.compile(
    r"The answer is:\s*(-?[\d,\.]+)",
    flags=re.MULTILINE | re.IGNORECASE
)


def match_format_exactly(prompts, completions, **kwargs):
    return [
        0 if match_format.search(response) is None else 3.0
        for response in completions
    ]


def match_format_approximately(prompts, completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion.lower()
        # Reward if contains "answer is" phrase
        if "the answer is" in response or "answer is:" in response:
            score += 1.5
        # Reward if contains any number
        if re.search(r'\d+', response):
            score += 0.5
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    responses = completions
    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    assert len(extracted_responses) == len(answer), \
        f"{extracted_responses} and {answer} have mismatching length"

    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        if guess == true_answer:
            score += 3.0
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0
            except:
                score -= 0.5
        scores.append(score)
    return scores


def make_check_numbers(show_conversation):
    '''Factory function to create check_numbers with show_conversation closure.'''
    def check_numbers(prompts, completions, answer, **kwargs):
        question = kwargs["question"]
        responses = completions

        extracted_responses = [
            guess.group(1) if (guess := match_numbers.search(r)) is not None else None
            for r in responses
        ]

        scores = []
        if show_conversation:
            print("START ============================")
            print(f"Question: {question[0]}")
            print(f"Answer: {answer[0]}")
            print(f"Response: {responses[0]}")
            print(f"Extracted: {extracted_responses[0]}")
            print("END ==============================")

        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0)
                continue
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip())
                scores.append(1.5 if guess == true_answer else 0.0)
            except:
                scores.append(0)
                continue
        return scores
    return check_numbers


# ============================================================================
# Evaluation Functions
# ============================================================================

def make_generate(total_generation_steps):
    '''Factory function to create generate with total_generation_steps closure.'''
    def generate(question, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None):
        '''Given prompt, generates text.'''
        if isinstance(question, str):
            input_batch = [
                TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=question,
                ),
            ]
        else:
            input_batch = [
                TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=q,
                )
                for q in question
            ]

        out_data = sampler(
            input_strings=input_batch,
            max_generation_steps=total_generation_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            echo=False,
            seed=seed if seed is not None else None,
        )

        output = out_data.text
        if isinstance(question, str):
            return output[0]
        return output
    return generate


def make_evaluate(total_generation_steps):
    '''Factory function to create evaluate with generate dependency.'''
    generate = make_generate(total_generation_steps)

    def evaluate(
        dataset,
        sampler,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_passes=1,
        corr_lst=False,
        make_lst=False,
    ):
        '''Computes accuracy and percentage of outputs matching the format.'''
        response_lst = []
        corr = 0
        partially_corr = 0
        corr_format = 0
        total = 0

        for batch in tqdm(dataset):
            answers = batch["answer"]
            questions = batch["question"]

            multiple_call_responses = [[] for _ in range(len(questions))]
            for p in range(num_passes):
                responses = generate(
                    questions, sampler, temperature, top_k, top_p, seed=p
                )
                for idx, response in enumerate(responses):
                    multiple_call_responses[idx].append(response)

            for question, multiple_call_response, answer in zip(
                questions, multiple_call_responses, answers
            ):
                corr_ctr_per_question = 0
                partially_corr_per_question = 0
                corr_format_per_question = 0

                for response in multiple_call_response:
                    extracted_response = (
                        guess.group(1)
                        if (guess := match_numbers.search(response)) is not None
                        else "-1000000"
                    )
                    try:
                        if float(extracted_response.strip()) == float(answer.strip()):
                            corr_ctr_per_question += 1

                        ratio = float(extracted_response.strip()) / float(answer.strip())
                        if ratio >= 0.9 and ratio <= 1.1:
                            partially_corr_per_question += 1
                    except:
                        print("SKIPPED")

                    if match_format.search(response) is not None:
                        corr_format_per_question += 1

                    if (
                        corr_ctr_per_question > 0
                        and partially_corr_per_question > 0
                        and corr_format_per_question > 0
                    ):
                        break

                if corr_ctr_per_question > 0:
                    corr += 1
                    if corr_lst and make_lst:
                        response_lst.append((question, answer, multiple_call_response))
                else:
                    if not corr_lst and make_lst:
                        response_lst.append((question, answer, multiple_call_response))

                if partially_corr_per_question > 0:
                    partially_corr += 1
                if corr_format_per_question > 0:
                    corr_format += 1

                total += 1
                if total % 10 == 0:
                    print(
                        f"===> {corr=}, {total=}, {corr / total * 100=}, "
                        f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
                    )

        to_return = (
            corr,
            total,
            corr / total * 100,
            partially_corr / total * 100,
            corr_format / total * 100,
        )
        if make_lst:
            return to_return, response_lst
        return to_return
    return evaluate


# ============================================================================
# CLI Definition
# ============================================================================

@click.command()
# Data options
@click.option('--train-data-dir', default='./data/train', show_default=True,
              help='Training data directory')
@click.option('--test-data-dir', default='./data/test', show_default=True,
              help='Test data directory')
@click.option('--train-fraction', default=1.0, show_default=True,
              help='Fraction of training data to use')
@click.option('--data-source', type=click.Choice(['tfds', 'kaggle']), default='kaggle',
              show_default=True, help='Data source')
# LoRA options
@click.option('--lora-rank', default=64, show_default=True, help='LoRA rank')
@click.option('--lora-alpha', default=64.0, show_default=True, help='LoRA alpha')
# GRPO options
@click.option('--max-prompt-length', default=128, show_default=True, help='Maximum prompt length')
@click.option('--total-generation-steps', default=256, show_default=True,
              help='Total generation steps')
@click.option('--temperature', default=0.9, show_default=True, help='Sampling temperature')
@click.option('--top-p', default=1.0, show_default=True, help='Top-p sampling')
@click.option('--top-k', default=50, show_default=True, help='Top-k sampling')
@click.option('--num-generations', default=2, show_default=True,
              help='Number of generations per prompt')
@click.option('--num-iterations', default=1, show_default=True,
              help='Number of iterations per batch')
@click.option('--beta', default=0.08, show_default=True, help='KL divergence penalty coefficient')
@click.option('--epsilon', default=0.2, show_default=True, help='PPO clipping epsilon')
# Training options
@click.option('--train-micro-batch-size', default=1, show_default=True,
              help='Training micro batch size')
@click.option('--num-batches', default=50, show_default=True, help='Number of training batches')
@click.option('--num-test-batches', default=30, show_default=True, help='Number of test batches')
@click.option('--eval-every-n-steps', default=10, show_default=True, help='Evaluate every N steps')
@click.option('--num-epochs', default=1, show_default=True, help='Number of training epochs')
# Optimizer options
@click.option('--learning-rate', default=3e-6, show_default=True, help='Learning rate')
@click.option('--b1', default=0.9, show_default=True, help='Adam beta1')
@click.option('--b2', default=0.99, show_default=True, help='Adam beta2')
@click.option('--weight-decay', default=0.1, show_default=True, help='Weight decay')
@click.option('--max-grad-norm', default=0.1, show_default=True,
              help='Max gradient norm for clipping')
# Checkpoint options
@click.option('--save-interval-steps', default=500, show_default=True,
              help='Save checkpoint every N steps')
@click.option('--max-to-keep', default=4, show_default=True, help='Maximum checkpoints to keep')
# Model options
@click.option('--model-family', type=click.Choice(['gemma3']), default='gemma3', show_default=True,
              help='Model family')
@click.option('--model-version', default='gemma3-1b-it', show_default=True, help='Model version')
# CPU offloading
@click.option('--offload-to-cpu/--no-offload-to-cpu', default=False, show_default=True,
              help='Offload tensors to CPU to save GPU memory')
# Conversation display
@click.option('--show-conversation/--dont-show-conversation', default=False, show_default=True,
              help='Show LLM conversation details during training')
def main(
    train_data_dir, test_data_dir, train_fraction, data_source,
    lora_rank, lora_alpha,
    max_prompt_length, total_generation_steps, temperature, top_p, top_k,
    num_generations, num_iterations, beta, epsilon,
    train_micro_batch_size, num_batches, num_test_batches, eval_every_n_steps, num_epochs,
    learning_rate, b1, b2, weight_decay, max_grad_norm,
    save_interval_steps, max_to_keep,
    model_family, model_version,
    offload_to_cpu,
    show_conversation
):
    '''GRPO training for Gemma3-1b on GSM8K math reasoning benchmark.'''

    # Create Trek for logging
    trek = Trek()

    with trek:
        print("=" * 60)
        print(f"{Color.BOLD}{Color.CYAN}GRPO Training - {model_version} on GSM8K{Color.END}")
        print("=" * 60)
        print()

        # Write hyperparameters to jsonla
        trek.hyperparameters_writer.write({
            'learning_rate': learning_rate,
            'beta': beta,
            'epsilon': epsilon,
            'temperature': temperature,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'num_generations': num_generations,
            'num_iterations': num_iterations,
            'train_micro_batch_size': train_micro_batch_size,
            'num_batches': num_batches,
            'num_test_batches': num_test_batches,
            'eval_every_n_steps': eval_every_n_steps,
            'num_epochs': num_epochs,
            'b1': b1,
            'b2': b2,
            'weight_decay': weight_decay,
            'max_grad_norm': max_grad_norm,
            'save_interval_steps': save_interval_steps,
            'max_to_keep': max_to_keep,
            'model_family': model_family,
            'model_version': model_version,
            'offload_to_cpu': offload_to_cpu,
            'max_prompt_length': max_prompt_length,
            'total_generation_steps': total_generation_steps,
            'top_p': top_p,
            'top_k': top_k,
            'train_fraction': train_fraction,
            'data_source': data_source,
            'train_data_dir': train_data_dir,
            'test_data_dir': test_data_dir,
        })

        # Create temporary directories for this training session
        with create_temp_folder(prefix='grpo_training_') as temp_base_dir:
            # Setup temp directories
            intermediate_ckpt_dir = temp_base_dir / "intermediate_ckpt"
            ckpt_dir = temp_base_dir / "ckpts"
            tensorboard_dir = temp_base_dir / "tensorboard" / "grpo"

            # Create directories
            intermediate_ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)

            # Convert to strings for compatibility with libraries expecting string paths
            intermediate_ckpt_dir = str(intermediate_ckpt_dir)
            ckpt_dir = str(ckpt_dir)
            tensorboard_dir = str(tensorboard_dir)

            print(f"{Color.YELLOW}Temporary directories created:{Color.END}")
            print(f"  Base: {temp_base_dir}")
            print(f"  Checkpoints: {ckpt_dir}")
            print(f"  TensorBoard: {tensorboard_dir}")
            print()

            _run_training(
                trek,
                str(intermediate_ckpt_dir), str(ckpt_dir), str(tensorboard_dir), train_data_dir,
                test_data_dir, train_fraction, data_source, lora_rank, lora_alpha,
                max_prompt_length, total_generation_steps, temperature, top_p, top_k,
                num_generations, num_iterations, beta, epsilon, train_micro_batch_size,
                num_batches, num_test_batches, eval_every_n_steps, num_epochs, learning_rate, b1,
                b2, weight_decay, max_grad_norm, save_interval_steps, max_to_keep, model_family,
                model_version, offload_to_cpu, show_conversation,
            )

            # Cleanup message
            print(f"\n{Color.YELLOW}Cleaning up temporary directories...{Color.END}")
            print(f"  Removing: {temp_base_dir}")


def _run_training(
    trek,
    intermediate_ckpt_dir, ckpt_dir, tensorboard_dir,
    train_data_dir, test_data_dir, train_fraction, data_source,
    lora_rank, lora_alpha,
    max_prompt_length, total_generation_steps, temperature, top_p, top_k,
    num_generations, num_iterations, beta, epsilon,
    train_micro_batch_size, num_batches, num_test_batches, eval_every_n_steps, num_epochs,
    learning_rate, b1, b2, weight_decay, max_grad_norm,
    save_interval_steps, max_to_keep,
    model_family, model_version,
    offload_to_cpu,
    show_conversation
):
    '''Run the actual training with the provided configuration.'''

    # ========================================================================
    # Device Detection and Configuration
    # ========================================================================
    print(f"{Color.BOLD}Detecting devices and configuring mesh...{Color.END}")

    # Automatically detect available devices and configure mesh
    num_devices = len(jax.devices())
    print(f"  Detected {num_devices} JAX device(s): {jax.devices()}")

    # Configure mesh based on available devices
    # For FSDP (Fully Sharded Data Parallel) and TP (Tensor Parallel)
    if num_devices >= 8:
        mesh_shape = (1, 8)
        print(f"  Using mesh shape {mesh_shape} (1 FSDP, 8 TP)")
    elif num_devices >= 4:
        mesh_shape = (1, 4)
        print(f"  Using mesh shape {mesh_shape} (1 FSDP, 4 TP)")
    elif num_devices >= 2:
        mesh_shape = (1, 2)
        print(f"  Using mesh shape {mesh_shape} (1 FSDP, 2 TP)")
    else:
        mesh_shape = (1, 1)
        print(f"  {Color.YELLOW}WARNING: Only 1 device available. "
              f"Training will be slow!{Color.END}")
        print(f"  Using mesh shape {mesh_shape} (no parallelism)")

    mesh_config = [mesh_shape, ("fsdp", "tp")]

    # Computed values
    max_steps = int(num_batches * num_iterations * train_fraction * num_epochs)
    warmup_steps = int(0.1 * max_steps)
    steps_per_iteration = int(num_batches * train_fraction)

    # Override save_interval_steps to save after each iteration
    save_interval_steps = steps_per_iteration

    # Inference generation configs
    generation_configs = {
        "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
        "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
    }

    print(f"  - Training steps: {max_steps}")
    print(f"  - LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - GRPO beta: {beta}, epsilon: {epsilon}")
    print(f"  - Num batches: {num_batches}, test batches: {num_test_batches}")
    print(f"{Color.GREEN}✓ Configuration set{Color.END}\n")

    # ========================================================================
    # Load datasets
    # ========================================================================
    print(f"{Color.BOLD}Loading GSM8K datasets...{Color.END}")
    try:
        print(f"  Using data source: {data_source}")

        dataset = get_dataset(train_data_dir, "train", data_source).batch(train_micro_batch_size)[
            :num_batches
        ]

        if train_fraction == 1.0:
            train_dataset = dataset.repeat(num_epochs)
            val_dataset = None
        else:
            train_dataset = dataset[: int(len(dataset) * train_fraction)]
            train_dataset = train_dataset.repeat(num_epochs)
            val_dataset = dataset[int(len(dataset) * train_fraction) :].repeat(num_epochs)

        test_dataset = get_dataset(test_data_dir, "test",
                                   data_source).batch(train_micro_batch_size)[:num_test_batches]

        dataset_lengths = (
            len(train_dataset),
            len(val_dataset) if val_dataset is not None else 0,
            len(test_dataset),
        )
        print(f"  Dataset sizes (train, val, test): {dataset_lengths}")
        print(f"{Color.GREEN}✓ Datasets loaded{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Dataset loading failed: {e}{Color.END}")
        return

    # ========================================================================
    # Authenticate with Kaggle
    # ========================================================================
    print(f"{Color.BOLD}Authenticating with Kaggle...{Color.END}")
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        print("  Note: Kaggle credentials not found in environment")
        print("  You may need to run: kagglehub.login()")
    print(f"{Color.GREEN}✓ Kaggle authentication ready{Color.END}\n")

    # ========================================================================
    # Download model from Kaggle
    # ========================================================================
    print(f"{Color.BOLD}Downloading Gemma3-1b-it from Kaggle...{Color.END}")
    try:
        model_path = {"gemma3": "google/gemma-3/flax/"}
        model_family = "gemma3"
        model_version = "gemma3-1b-it"

        print(f"  Model: {model_path[model_family]}{model_version}")
        kaggle_ckpt_path = kagglehub.model_download(
            f"{model_path[model_family]}{model_version}"
        )
        print(f"  Downloaded to: {kaggle_ckpt_path}")
        print(f"{Color.GREEN}✓ Model downloaded{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Model download failed: {e}{Color.END}")
        print("  Make sure you have accepted the Gemma license on Kaggle")
        return

    # ========================================================================
    # Prepare checkpoint directories
    # ========================================================================
    print(f"{Color.BOLD}Preparing checkpoint directories...{Color.END}")
    try:
        # Clean checkpoint directories
        if os.path.exists(intermediate_ckpt_dir):
            shutil.rmtree(intermediate_ckpt_dir)
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        # Create directories
        os.makedirs(intermediate_ckpt_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Note: Gemma3 loads directly from safetensors, no conversion needed
        print(f"  Gemma3 will load directly from safetensors at: {kaggle_ckpt_path}")

        print(f"{Color.GREEN}✓ Directories prepared{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Directory preparation failed: {e}{Color.END}")
        return

    # ========================================================================
    # Load reference model
    # ========================================================================
    print(f"{Color.BOLD}Loading reference model (Gemma3-1b-it)...{Color.END}")
    try:
        if model_family == "gemma3":
            # Load from Kaggle Orbax checkpoint
            # Path structure: kaggle_ckpt_path/gemma3-1b-it/
            checkpoint_path = os.path.join(kaggle_ckpt_path, model_version)
            print(f"  Loading from checkpoint: {checkpoint_path}")
            ref_model, mesh, model_config = get_gemma_ref_model(
                ckpt_path=checkpoint_path,
                mesh_config=mesh_config
            )
        print(f"{Color.GREEN}✓ Reference model loaded{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Reference model loading failed: {e}{Color.END}")
        return

    # ========================================================================
    # Apply LoRA to create policy model
    # ========================================================================
    print(f"{Color.BOLD}Applying LoRA to create policy model...{Color.END}")
    try:
        lora_policy = get_lora_model(ref_model, mesh=mesh, lora_rank=lora_rank,
                                     lora_alpha=lora_alpha)
        # print("  Policy model structure:")
        # nnx.display(lora_policy)
        print(f"{Color.GREEN}✓ LoRA policy model created{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ LoRA application failed: {e}{Color.END}")
        return

    # ========================================================================
    # Load tokenizer
    # ========================================================================
    print(f"{Color.BOLD}Loading tokenizer...{Color.END}")
    try:
        if model_family == "gemma3":
            tokenizer = tokenizer_lib.Tokenizer(
                tokenizer_path=os.path.join(kaggle_ckpt_path, "tokenizer.model")
            )
        print(f"{Color.GREEN}✓ Tokenizer loaded{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Tokenizer loading failed: {e}{Color.END}")
        return

    # Create evaluation and reward functions with closures
    evaluate = make_evaluate(total_generation_steps)
    check_numbers = make_check_numbers(show_conversation)

    # ========================================================================
    # Create sampler for evaluation
    # ========================================================================
    print(f"{Color.BOLD}Creating sampler for pre-training evaluation...{Color.END}")
    try:
        sampler = sampler_lib.Sampler(
            transformer=lora_policy,
            tokenizer=tokenizer,
            cache_config=sampler_lib.CacheConfig(
                cache_size=max_prompt_length + total_generation_steps + 256,
                num_layers=model_config.num_layers,
                num_kv_heads=model_config.num_kv_heads,
                head_dim=model_config.head_dim,
            ),
        )
        print(f"{Color.GREEN}✓ Sampler created{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Sampler creation failed: {e}{Color.END}")
        return

    # ========================================================================
    # Pre-training evaluation
    # ========================================================================
    print(f"{Color.BOLD}Running pre-training evaluation on test set...{Color.END}")
    print("  (This may take a few minutes)")
    try:
        (corr, total, pre_train_accuracy, pre_train_partial_accuracy, pre_train_format_accuracy) = \
                                     evaluate(test_dataset, sampler, **generation_configs["greedy"])
        print(f"\n  Pre-training results:")
        print(f"    Correct: {corr}/{total}")
        print(f"    Accuracy: {pre_train_accuracy:.2f}%")
        print(f"    Partial accuracy: {pre_train_partial_accuracy:.2f}%")
        print(f"    Format accuracy: {pre_train_format_accuracy:.2f}%")
        print(f"{Color.GREEN}✓ Pre-training evaluation done{Color.END}\n")

        # Write pre-training results to Trek
        trek.results_writer.write({
            'phase': 'pre_training',
            'iteration': 0,
            'step': 0,
            'accuracy': pre_train_accuracy,
            'partial_accuracy': pre_train_partial_accuracy,
            'format_accuracy': pre_train_format_accuracy,
        })
    except Exception as e:
        print(f"{Color.RED}✗ Pre-training evaluation failed: {e}{Color.END}")
        print("  Cannot continue without successful pre-training evaluation")
        return

    # ========================================================================
    # Setup checkpointing and metrics logging
    # ========================================================================
    print(f"{Color.BOLD}Setting up checkpointing and metrics logging...{Color.END}")
    try:
        checkpointing_options = ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_steps, max_to_keep=max_to_keep
        )

        metrics_logging_options = metrics_logger.MetricsLoggerOptions(
            log_dir=tensorboard_dir, flush_every_n_steps=20
        )

        print(f"  Checkpoint dir: {ckpt_dir}")
        print(f"  Save interval: every {save_interval_steps} steps")
        print(f"  Max checkpoints to keep: {max_to_keep}")
        print(f"  Tensorboard logs: {tensorboard_dir}")
        print(f"{Color.GREEN}✓ Checkpointing configured{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Checkpointing setup failed: {e}{Color.END}")
        return

    # ========================================================================
    # Setup optimizer and learning rate schedule
    # ========================================================================
    print(f"{Color.BOLD}Setting up optimizer and learning rate schedule...{Color.END}")
    try:
        optimizer = optax.adamw(
            learning_rate=optax.schedules.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
        )

        if max_grad_norm is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(max_norm=max_grad_norm),
                optimizer,
            )

        print(f"  Optimizer: AdamW")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Grad clipping: {max_grad_norm}")
        print(f"{Color.GREEN}✓ Optimizer configured{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Optimizer setup failed: {e}{Color.END}")
        return

    # ========================================================================
    # Create RL cluster configuration
    # ========================================================================
    print(f"{Color.BOLD}Creating RL cluster configuration...{Color.END}")
    try:
        cluster_config = rl_cluster_lib.ClusterConfig(
            role_to_mesh={
                rl_cluster_lib.Role.ACTOR: mesh,
                rl_cluster_lib.Role.REFERENCE: mesh,
                rl_cluster_lib.Role.ROLLOUT: mesh,
            },
            rollout_engine='vanilla',
            offload_to_cpu=False,
            training_config=rl_cluster_lib.RLTrainingConfig(
                actor_optimizer=optimizer,
                eval_every_n_steps=eval_every_n_steps,
                max_steps=max_steps,
                mini_batch_size=train_micro_batch_size,
                train_micro_batch_size=train_micro_batch_size,
                metrics_logging_options=metrics_logging_options,
                checkpoint_root_directory=ckpt_dir,
                checkpointing_options=checkpointing_options,
            ),
            rollout_config=base_rollout.RolloutConfig(
                max_tokens_to_generate=total_generation_steps,
                max_prompt_length=max_prompt_length,
                kv_cache_size=max_prompt_length + total_generation_steps + 256,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
        )

        grpo_config = GRPOConfig(
            num_generations=num_generations,
            num_iterations=num_iterations,
            beta=beta,
            epsilon=epsilon,
        )

        print("  RL Cluster configured with:")
        print(f"    - Actor, Reference, and Rollout roles")
        print(f"    - Rollout engine: vanilla")
        print(f"    - Max training steps: {max_steps}")
        print(f"{Color.GREEN}✓ RL cluster configured{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ RL cluster configuration failed: {e}{Color.END}")
        return

    # ========================================================================
    # Initialize RL cluster and GRPO learner
    # ========================================================================
    print(f"{Color.BOLD}Initializing RL cluster and GRPO learner...{Color.END}")
    try:
        rl_cluster = rl_cluster_lib.RLCluster(
            actor=lora_policy,
            reference=ref_model,
            tokenizer=tokenizer,
            cluster_config=cluster_config,
        )

        grpo_trainer = GRPOLearner(
            rl_cluster=rl_cluster,
            reward_fns=[
                match_format_exactly,
                match_format_approximately,
                check_answer,
                check_numbers,
            ],
            grpo_config=grpo_config,
        )

        print("  GRPO Learner initialized with 4 reward functions")
        print(f"{Color.GREEN}✓ GRPO trainer ready{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ GRPO initialization failed: {e}{Color.END}")
        return

    # ========================================================================
    # Run GRPO training
    # ========================================================================
    print(f"{Color.BOLD}Starting GRPO training...{Color.END}")
    print("  Note: First training step may take up to 5 minutes")
    print("  This is a long-running process. Press Ctrl+C to stop.")
    print()

    try:
        with mesh:
            grpo_trainer.train(train_dataset)
        print("\n✓ Training finished\n")
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user\n")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        print("  Training encountered an error")
        return

    # ========================================================================
    # Evaluate after each iteration
    # ========================================================================
    print(f"{Color.BOLD}Evaluating model after each iteration...{Color.END}")

    iteration_results = []
    steps_per_iteration = num_batches * train_fraction

    try:
        abs_params = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            nnx.state(lora_policy, nnx.LoRAParam),
        )
        checkpointer = ocp.StandardCheckpointer()

        # Evaluate after each iteration
        for iteration in range(1, num_iterations + 1):
            iteration_step = int(iteration * steps_per_iteration)
            checkpoint_path = os.path.join(ckpt_dir, "actor", str(iteration_step), "model_params")

            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                print(f"  Warning: Checkpoint for iteration {iteration} "
                      f"(step {iteration_step}) not found, skipping")
                continue

            print(f"\n  Evaluating iteration {iteration}/{num_iterations} "
                  f"(step {iteration_step})...")

            # Load checkpoint for this iteration
            iteration_params = checkpointer.restore(checkpoint_path, target=abs_params)
            nnx.update(
                lora_policy,
                jax.tree.map(
                    lambda a, b: b,
                    nnx.state(lora_policy, nnx.LoRAParam),
                    iteration_params,
                ),
            )

            # Create sampler for evaluation
            sampler = sampler_lib.Sampler(
                transformer=lora_policy,
                tokenizer=tokenizer,
                cache_config=sampler_lib.CacheConfig(
                    cache_size=max_prompt_length + total_generation_steps + 256,
                    num_layers=model_config.num_layers,
                    num_kv_heads=model_config.num_kv_heads,
                    head_dim=model_config.head_dim,
                ),
            )

            # Evaluate
            (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
                test_dataset,
                sampler,
                **generation_configs["greedy"],
            )

            result_data = {
                'phase': 'post_training',
                'iteration': iteration,
                'step': iteration_step,
                'accuracy': accuracy,
                'partial_accuracy': partial_accuracy,
                'format_accuracy': format_accuracy,
            }

            iteration_results.append(result_data)

            # Write to Trek
            trek.results_writer.write(result_data)

            print(f"    Accuracy: {accuracy:.2f}%")

        print(f"{Color.GREEN}✓ Per-iteration evaluation done{Color.END}\n")

    except Exception as e:
        print(f"{Color.RED}✗ Per-iteration evaluation failed: {e}{Color.END}")
        print("  Could not complete per-iteration evaluation")
        # Continue anyway to show final results
        pass

    # ========================================================================
    # Display results summary
    # ========================================================================
    print("=" * 60)
    print(f"{Color.BOLD}{Color.GREEN}GRPO Training Complete!{Color.END}")
    print("=" * 60)
    print()

    print(f"{Color.BOLD}Training Progress Summary:{Color.END}")
    print(f"  Pre-training accuracy: {pre_train_accuracy:.2f}%")

    if iteration_results:
        for result in iteration_results:
            improvement = result['accuracy'] - pre_train_accuracy
            print(f"  After iteration {result['iteration']} (step {result['step']}): "
                  f"{result['accuracy']:.2f}% (improvement: {improvement:+.2f}%)")

        # Final results
        final_result = iteration_results[-1]
        final_improvement = final_result['accuracy'] - pre_train_accuracy
        print()
        print(f"{Color.BOLD}Final Results:{Color.END}")
        print(f"  Final accuracy: {final_result['accuracy']:.2f}%")
        print(f"  Total improvement: {final_improvement:+.2f}%")
    else:
        print(f"  {Color.YELLOW}No iteration results available{Color.END}")
    print()
    print(f"{Color.BOLD}Checkpoints saved to:{Color.END} {ckpt_dir}")
    print(f"{Color.BOLD}TensorBoard logs:{Color.END} {tensorboard_dir}")
    print()

    # Return results for hyperparameter search
    if iteration_results:
        final_result = iteration_results[-1]
        return {
            'pre_train_accuracy': pre_train_accuracy,
            'pre_train_partial_accuracy': pre_train_partial_accuracy,
            'pre_train_format_accuracy': pre_train_format_accuracy,
            'post_train_accuracy': final_result['accuracy'],
            'post_train_partial_accuracy': final_result['partial_accuracy'],
            'post_train_format_accuracy': final_result['format_accuracy'],
            'improvement': final_result['accuracy'] - pre_train_accuracy,
            'iteration_results': iteration_results,
        }
    else:
        # Fallback if no iteration results
        return {
            'pre_train_accuracy': pre_train_accuracy,
            'pre_train_partial_accuracy': pre_train_partial_accuracy,
            'pre_train_format_accuracy': pre_train_format_accuracy,
            'post_train_accuracy': pre_train_accuracy,  # No improvement
            'post_train_partial_accuracy': pre_train_partial_accuracy,
            'post_train_format_accuracy': pre_train_format_accuracy,
            'improvement': 0.0,
            'iteration_results': [],
        }


if __name__ == "__main__":
    main()
