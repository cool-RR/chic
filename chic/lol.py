#!/usr/bin/env python
'''
GRPO Demo - Training LLMs on GSM8K math reasoning benchmark
Adapted from Tunix GRPO demo notebook
'''

import functools
import gc
import os
import re
import csv
import shutil
import sys
import tempfile
import pathlib
from contextlib import contextmanager
import textwrap
import datetime as datetime_module
import time

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
import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

from chic.trekking import Trek
from chic.misc.datetime_tools import format_timedelta
from chic.misc.temp_file_tools import create_temp_folder
from chic.model_brand import ModelBrand, MODEL_NAMES


# ANSI color codes for terminal formatting
class Color:
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'


class DotIterator:
    '''Iterator that prints dots for progress indication in non-TTY environments.'''
    def __init__(self, iterator, start_message=None):
        self.iterator = iterator
        self.start_message = start_message

    def __iter__(self):
        if self.start_message is not None:
            print(self.start_message, end='')
            sys.stdout.flush()
        for item in self.iterator:
            yield item
            print('.', end='')
            sys.stdout.flush()


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

    def evaluate(dataset, sampler, temperature=0.7, top_k=50, top_p=0.95, n_passes=1,
                 corr_lst=False, make_lst=False):
        '''Computes accuracy and percentage of outputs matching the format.'''
        response_lst = []
        corr = 0
        partially_corr = 0
        corr_format = 0
        total = 0

        # Use tqdm only if stdout is a tty, with viola's styling
        if sys.stdout.isatty():
            dataset_iter = tqdm.tqdm(
                dataset,
                colour='cyan',
                ascii='_▒▓█',
                ncols=70,
                file=sys.__stdout__,
            )
        else:
            dataset_iter = DotIterator(dataset, start_message='Evaluating')

        for batch in dataset_iter:
            answers = batch["answer"]
            questions = batch["question"]

            multiple_call_responses = [[] for _ in range(len(questions))]
            for p in range(n_passes):
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
@click.option('--n-generations', default=2, show_default=True,
              help='Number of generations per prompt')
@click.option('--n-iterations', default=1, show_default=True,
              help='Number of iterations per batch')
@click.option('--beta', default=0.08, show_default=True, help='KL divergence penalty coefficient')
@click.option('--epsilon', default=0.2, show_default=True, help='PPO clipping epsilon')
# Training options
@click.option('--train-micro-batch-size', default=1, show_default=True,
              help='Training micro batch size')
@click.option('--n-batches', default=50, show_default=True, help='Number of training batches')
@click.option('--n-test-batches', default=30, show_default=True, help='Number of test batches')
@click.option('--eval-every-n-steps', default=10, show_default=True, help='Evaluate every N steps')
@click.option('--n-epochs', default=1, show_default=True, help='Number of training epochs')
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
@click.option('--model', type=click.Choice(MODEL_NAMES), default='gemma3-1b-it',
              show_default=True, help='Model to train')
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
    n_generations, n_iterations, beta, epsilon,
    train_micro_batch_size, n_batches, n_test_batches, eval_every_n_steps, n_epochs,
    learning_rate, b1, b2, weight_decay, max_grad_norm,
    save_interval_steps, max_to_keep,
    model,
    offload_to_cpu,
    show_conversation
):
    '''GRPO training for various LLMs on GSM8K math reasoning benchmark.'''

    # Create Trek for logging
    trek = Trek()

    with trek:
        print("=" * 60)
        print(f"{Color.BOLD}{Color.CYAN}GRPO Training - {model} on GSM8K{Color.END}")
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
            'n_generations': n_generations,
            'n_iterations': n_iterations,
            'train_micro_batch_size': train_micro_batch_size,
            'n_batches': n_batches,
            'n_test_batches': n_test_batches,
            'eval_every_n_steps': eval_every_n_steps,
            'n_epochs': n_epochs,
            'b1': b1,
            'b2': b2,
            'weight_decay': weight_decay,
            'max_grad_norm': max_grad_norm,
            'save_interval_steps': save_interval_steps,
            'max_to_keep': max_to_keep,
            'model': model,
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
                n_generations, n_iterations, beta, epsilon, train_micro_batch_size,
                n_batches, n_test_batches, eval_every_n_steps, n_epochs, learning_rate, b1,
                b2, weight_decay, max_grad_norm, save_interval_steps, max_to_keep, model,
                offload_to_cpu, show_conversation,
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
    n_generations, n_iterations, beta, epsilon,
    train_micro_batch_size, n_batches, n_test_batches, eval_every_n_steps, n_epochs,
    learning_rate, b1, b2, weight_decay, max_grad_norm,
    save_interval_steps, max_to_keep,
    model,
    offload_to_cpu,
    show_conversation
):
    '''Run the actual training with the provided configuration.'''

    # Track overall run time
    run_start_time = time.time()

    # ========================================================================
    # Phase 1/17: Device Detection and Configuration
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 1/17: Detecting devices and configuring mesh...{Color.END}")

    # Automatically detect available devices and configure mesh
    n_devices = len(jax.devices())
    print(f"  Detected {n_devices} JAX device(s): {jax.devices()}")

    # Configure mesh based on available devices
    # For FSDP (Fully Sharded Data Parallel) and TP (Tensor Parallel)
    if n_devices >= 8:
        mesh_shape = (1, 8)
        print(f"  Using mesh shape {mesh_shape} (1 FSDP, 8 TP)")
    elif n_devices >= 4:
        mesh_shape = (1, 4)
        print(f"  Using mesh shape {mesh_shape} (1 FSDP, 4 TP)")
    elif n_devices >= 2:
        mesh_shape = (1, 2)
        print(f"  Using mesh shape {mesh_shape} (1 FSDP, 2 TP)")
    else:
        mesh_shape = (1, 1)
        print(f"  {Color.YELLOW}WARNING: Only 1 device available. "
              f"Training will be slow!{Color.END}")
        print(f"  Using mesh shape {mesh_shape} (no parallelism)")

    mesh_config = [mesh_shape, ("fsdp", "tp")]

    # Computed values
    max_steps = int(n_batches * n_iterations * train_fraction * n_epochs)
    warmup_steps = int(0.1 * max_steps)
    steps_per_iteration = int(n_batches * train_fraction)

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
    print(f"  - Num batches: {n_batches}, test batches: {n_test_batches}")
    phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
    print(f"{Color.GREEN}✓ Phase 1/17 complete "
          f"[{format_timedelta(phase_duration)}]{Color.END}\n")

    # ========================================================================
    # Phase 2/17: Load datasets
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 2/17: Loading GSM8K datasets...{Color.END}")
    try:
        print(f"  Using data source: {data_source}")

        dataset = get_dataset(train_data_dir, "train", data_source).batch(train_micro_batch_size)[
            :n_batches
        ]

        if train_fraction == 1.0:
            train_dataset = dataset.repeat(n_epochs)
            val_dataset = None
        else:
            train_dataset = dataset[: int(len(dataset) * train_fraction)]
            train_dataset = train_dataset.repeat(n_epochs)
            val_dataset = dataset[int(len(dataset) * train_fraction) :].repeat(n_epochs)

        test_dataset = get_dataset(test_data_dir, "test",
                                   data_source).batch(train_micro_batch_size)[:n_test_batches]

        dataset_lengths = (
            len(train_dataset),
            len(val_dataset) if val_dataset is not None else 0,
            len(test_dataset),
        )
        print(f"  Dataset sizes (train, val, test): {dataset_lengths}")
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 2/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 2/17 failed: {e}{Color.END}")
        sys.exit(1)

    # ========================================================================
    # Phase 3/17: Authenticate with Kaggle
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 3/17: Authenticating with Kaggle...{Color.END}")
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        print("  Note: Kaggle credentials not found in environment")
        print("  You may need to run: kagglehub.login()")
    phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
    print(f"{Color.GREEN}✓ Phase 3/17 complete "
          f"[{format_timedelta(phase_duration)}]{Color.END}\n")

    # ========================================================================
    # Phase 4/17: Download model from Kaggle
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 4/17: Downloading {model} from Kaggle...{Color.END}")
    try:
        model_brand = ModelBrand.get_by_name(model)

        print(f"  Model: {model_brand.full_kaggle_path}")
        kaggle_ckpt_path = kagglehub.model_download(model_brand.full_kaggle_path)
        print(f"  Downloaded to: {kaggle_ckpt_path}")
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 4/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 4/17 failed: {e}{Color.END}")
        print("  Make sure you have accepted the model license on Kaggle")
        sys.exit(1)

    # ========================================================================
    # Phase 5/17: Prepare checkpoint directories
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 5/17: Preparing checkpoint directories...{Color.END}")
    try:
        # Clean checkpoint directories
        if os.path.exists(intermediate_ckpt_dir):
            shutil.rmtree(intermediate_ckpt_dir)
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        # Create directories
        os.makedirs(intermediate_ckpt_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        print(f"  Model will load from: {kaggle_ckpt_path}")

        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 5/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 5/17 failed: {e}{Color.END}")
        sys.exit(1)

    # ========================================================================
    # Phase 6/17: Load reference model
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 6/17: Loading reference model ({model})...{Color.END}")
    try:
        ref_model, mesh, model_config = model_brand.load_model(
            ckpt_path=kaggle_ckpt_path,
            mesh_config=mesh_config
        )
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 6/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 6/17 failed: {e}{Color.END}")
        sys.exit(1)

    # ========================================================================
    # Phase 7/17: Apply LoRA to create policy model
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 7/17: Applying LoRA to create policy model...{Color.END}")
    try:
        lora_policy = get_lora_model(ref_model, mesh=mesh, lora_rank=lora_rank,
                                     lora_alpha=lora_alpha)
        # print("  Policy model structure:")
        # nnx.display(lora_policy)
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 7/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 7/17 failed: {e}{Color.END}")
        sys.exit(1)

    # ========================================================================
    # Phase 8/17: Load tokenizer
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 8/17: Loading tokenizer...{Color.END}")
    try:
        tokenizer = tokenizer_lib.Tokenizer(
            tokenizer_path=model_brand.get_tokenizer_path(kaggle_ckpt_path)
        )
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 8/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 8/17 failed: {e}{Color.END}")
        sys.exit(1)

    # Create evaluation and reward functions with closures
    evaluate = make_evaluate(total_generation_steps)
    check_numbers = make_check_numbers(show_conversation)

    # ========================================================================
    # Phase 9/17: Create sampler for evaluation
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 9/17: Creating sampler for pre-training evaluation...{Color.END}")
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
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 9/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 9/17 failed: {e}{Color.END}")
        sys.exit(1)

    # ========================================================================
    # Phase 10/17: Pre-training evaluation
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 10/17: Running pre-training evaluation on test set...{Color.END}")
    print("  (This may take a few minutes)")
    try:
        (corr, total, pre_train_accuracy, pre_train_partial_accuracy, pre_train_format_accuracy) = \
                                     evaluate(test_dataset, sampler, **generation_configs["greedy"])
        print(f"\n  Pre-training results:")
        print(f"    Correct: {corr}/{total}")
        print(f"    Accuracy: {pre_train_accuracy:.2f}%")
        print(f"    Partial accuracy: {pre_train_partial_accuracy:.2f}%")
        print(f"    Format accuracy: {pre_train_format_accuracy:.2f}%")

        # Write pre-training results to Trek
        trek.results_writer.write({
            'phase': 'pre_training',
            'iteration': 0,
            'step': 0,
            'accuracy': pre_train_accuracy,
            'partial_accuracy': pre_train_partial_accuracy,
            'format_accuracy': pre_train_format_accuracy,
        })
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 10/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 10/17 failed: {e}{Color.END}")
        print("  Cannot continue without successful pre-training evaluation")
        sys.exit(1)

    # ========================================================================
    # Phase 11/17: Setup checkpointing and metrics logging
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 11/17: Setting up checkpointing and metrics logging...{Color.END}")
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
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 11/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 11/17 failed: {e}{Color.END}")
        sys.exit(1)

    # ========================================================================
    # Phase 12/17: Setup optimizer and learning rate schedule
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 12/17: Setting up optimizer and learning rate schedule...{Color.END}")
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
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 12/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 12/17 failed: {e}{Color.END}")
        sys.exit(1)

    # ========================================================================
    # Phase 13/17: Create RL cluster configuration
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 13/17: Creating RL cluster configuration...{Color.END}")
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
            num_generations=n_generations,
            num_iterations=n_iterations,
            beta=beta,
            epsilon=epsilon,
        )

        print("  RL Cluster configured with:")
        print(f"    - Actor, Reference, and Rollout roles")
        print(f"    - Rollout engine: vanilla")
        print(f"    - Max training steps: {max_steps}")
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 13/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 13/17 failed: {e}{Color.END}")
        sys.exit(1)

    # ========================================================================
    # Phase 14/17: Initialize RL cluster and GRPO learner
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 14/17: Initializing RL cluster and GRPO learner...{Color.END}")
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
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 14/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Phase 14/17 failed: {e}{Color.END}")
        sys.exit(1)

    # ========================================================================
    # Phase 15/17: Run GRPO training
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 15/17: Starting GRPO training...{Color.END}")
    print("  Note: First training step may take up to 5 minutes")
    print("  This is a long-running process. Press Ctrl+C to stop.")
    print()

    try:
        with mesh:
            grpo_trainer.train(train_dataset)
        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"\n{Color.GREEN}✓ Phase 15/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user\n")
    except Exception as e:
        print(f"\n✗ Phase 15/17 failed: {e}")
        print("  Training encountered an error")
        sys.exit(1)

    # ========================================================================
    # Phase 16/17: Evaluate after each iteration
    # ========================================================================
    phase_start_time = time.time()
    print(f"{Color.BOLD}Phase 16/17: Evaluating model after each iteration...{Color.END}")

    iteration_results = []
    steps_per_iteration = n_batches * train_fraction

    try:
        abs_params = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            nnx.state(lora_policy, nnx.LoRAParam),
        )
        checkpointer = ocp.StandardCheckpointer()

        # Evaluate after each iteration
        for iteration in range(1, n_iterations + 1):
            iteration_step = int(iteration * steps_per_iteration)
            checkpoint_path = os.path.join(ckpt_dir, "actor", str(iteration_step), "model_params")

            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                print(f"  Warning: Checkpoint for iteration {iteration} "
                      f"(step {iteration_step}) not found, skipping")
                continue

            print(f"\n  Evaluating iteration {iteration}/{n_iterations} "
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

        phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
        print(f"{Color.GREEN}✓ Phase 16/17 complete "
              f"[{format_timedelta(phase_duration)}]{Color.END}\n")

    except Exception as e:
        print(f"{Color.RED}✗ Phase 16/17 failed: {e}{Color.END}")
        print("  Could not complete per-iteration evaluation")
        # Continue anyway to show final results
        pass

    # ========================================================================
    # Phase 17/17: Display results summary
    # ========================================================================
    phase_start_time = time.time()
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

    phase_duration = datetime_module.timedelta(seconds=time.time() - phase_start_time)
    print(f"{Color.GREEN}✓ Phase 17/17 complete "
          f"[{format_timedelta(phase_duration)}]{Color.END}\n")

    # Display total runtime
    total_runtime = datetime_module.timedelta(seconds=time.time() - run_start_time)
    print("=" * 60)
    print(f"{Color.BOLD}Total runtime: {format_timedelta(total_runtime)}{Color.END}")
    print("=" * 60)
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
