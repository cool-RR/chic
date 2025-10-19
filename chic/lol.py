#!/usr/bin/env python3
"""
GRPO Demo - Training Gemma2-2b-it on GSM8K math reasoning benchmark
Adapted from Tunix GRPO demo notebook
"""

import functools
import gc
import os
import re
import csv
import shutil


def main():
    """Main GRPO training function with staged progress reporting."""

    print("=" * 60)
    print("GRPO Training - Gemma2-2b-it on GSM8K")
    print("=" * 60)
    print()

    # ========================================================================
    # Phase 1: Imports
    # ========================================================================
    print("Phase 1: Loading dependencies...")
    try:
        from flax import nnx
        import grain
        import humanize
        import jax
        import jax.numpy as jnp
        import kagglehub
        import optax
        from orbax import checkpoint as ocp
        from pathlib import Path
        import qwix
        import tensorflow_datasets as tfds
        from tqdm.auto import tqdm
        from tunix.generate import sampler as sampler_lib
        from tunix.generate import tokenizer_adapter as tokenizer_lib
        from tunix.models.gemma import model as gemma_lib
        from tunix.models.gemma import params as params_lib
        from tunix.rl import rl_cluster as rl_cluster_lib
        from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
        from tunix.rl.rollout import base_rollout
        from tunix.sft import metrics_logger
        print("✓ Phase 1 complete: All dependencies loaded successfully\n")
    except ImportError as e:
        print(f"✗ Phase 1 failed: Missing dependency - {e}")
        print("Please install required packages:")
        print("  pip install flax grain jax optax orbax tensorflow_datasets")
        print("  pip install git+https://github.com/google/tunix")
        print("  pip install git+https://github.com/google/qwix")
        return

    # ========================================================================
    # Phase 2: Configuration
    # ========================================================================
    print("Phase 2: Setting up hyperparameters...")

    # ====== Data ======
    TRAIN_DATA_DIR = "./data/train"
    TEST_DATA_DIR = "./data/test"
    TRAIN_FRACTION = 1.0

    # ====== LoRA ======
    RANK = 64
    ALPHA = 64.0

    # ====== Sharding ======
    MESH = [(1, 4), ("fsdp", "tp")]

    # ====== GRPO ======
    MAX_PROMPT_LENGTH = 256
    TOTAL_GENERATION_STEPS = 768
    TEMPERATURE = 0.9
    TOP_P = 1.0
    TOP_K = 50
    NUM_GENERATIONS = 2
    NUM_ITERATIONS = 1
    BETA = 0.08
    EPSILON = 0.2

    # ====== Training ======
    TRAIN_MICRO_BATCH_SIZE = 1
    NUM_BATCHES = 3738
    NUM_TEST_BATCHES = 100
    EVAL_EVERY_N_STEPS = 10
    NUM_EPOCHS = 1
    MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

    # ====== AdamW, warmup, cosine scheduler ======
    LEARNING_RATE = 3e-6
    B1 = 0.9
    B2 = 0.99
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 0.1 * MAX_STEPS
    MAX_GRAD_NORM = 0.1

    # Checkpoint saving
    INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
    CKPT_DIR = "/tmp/content/ckpts/"
    SAVE_INTERVAL_STEPS = 500
    MAX_TO_KEEP = 4

    # ====== Inference ======
    GENERATION_CONFIGS = {
        "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
        "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
    }

    print(f"  - Training steps: {MAX_STEPS}")
    print(f"  - LoRA rank: {RANK}, alpha: {ALPHA}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - GRPO beta: {BETA}, epsilon: {EPSILON}")
    print("✓ Phase 2 complete: Configuration set\n")

    # ========================================================================
    # Phase 3: Define special tokens and templates
    # ========================================================================
    print("Phase 3: Setting up prompt templates...")

    reasoning_start = "<reasoning>"
    reasoning_end = "</reasoning>"
    solution_start = "<answer>"
    solution_end = "</answer>"

    SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

    TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""

    print("✓ Phase 3 complete: Prompt templates configured\n")

    # ========================================================================
    # Phase 4: Define utility functions
    # ========================================================================
    print("Phase 4: Defining utility functions...")

    def show_hbm_usage():
        """Displays memory usage per device."""
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
        src = Path(src)
        dst = Path(target_dir)
        for csv_file in src.glob("*.csv"):
            shutil.copy2(csv_file, dst / csv_file.name)
            print(f"  Copied {csv_file.name} → {dst/csv_file.name}")
        return target_dir

    def get_dataset(data_dir, split="train", source="tfds"):
        """Load and prepare dataset."""
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

    print("✓ Phase 4 complete: Utility functions defined\n")

    # ========================================================================
    # Phase 5: Load datasets
    # ========================================================================
    print("Phase 5: Loading GSM8K datasets...")
    try:
        # Default to tfds for automated runs
        source = "tfds"
        print(f"  Using data source: {source}")

        dataset = get_dataset(TRAIN_DATA_DIR, "train", source).batch(TRAIN_MICRO_BATCH_SIZE)[
            :NUM_BATCHES
        ]

        if TRAIN_FRACTION == 1.0:
            train_dataset = dataset.repeat(NUM_EPOCHS)
            val_dataset = None
        else:
            train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
            train_dataset = train_dataset.repeat(NUM_EPOCHS)
            val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

        test_dataset = get_dataset(TEST_DATA_DIR, "test", source).batch(TRAIN_MICRO_BATCH_SIZE)[
            :NUM_TEST_BATCHES
        ]

        dataset_lengths = (
            len(train_dataset),
            len(val_dataset) if val_dataset is not None else 0,
            len(test_dataset),
        )
        print(f"  Dataset sizes (train, val, test): {dataset_lengths}")
        print("✓ Phase 5 complete: Datasets loaded\n")
    except Exception as e:
        print(f"✗ Phase 5 failed: {e}")
        return

    # ========================================================================
    # Phase 6: Authenticate with Kaggle
    # ========================================================================
    print("Phase 6: Authenticating with Kaggle...")
    try:
        if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
            print("  Note: Kaggle credentials not found in environment")
            print("  You may need to run: kagglehub.login()")
        print("✓ Phase 6 complete: Kaggle authentication ready\n")
    except Exception as e:
        print(f"✗ Phase 6 failed: {e}")
        return

    # ========================================================================
    # Phase 7: Download model from Kaggle
    # ========================================================================
    print("Phase 7: Downloading Gemma2-2b-it from Kaggle...")
    try:
        model_path = {"gemma2": "google/gemma-2/flax/"}
        model_family = "gemma2"
        model_version = "gemma2-2b-it"

        print(f"  Model: {model_path[model_family]}{model_version}")
        kaggle_ckpt_path = kagglehub.model_download(
            f"{model_path[model_family]}{model_version}"
        )
        print(f"  Downloaded to: {kaggle_ckpt_path}")
        print("✓ Phase 7 complete: Model downloaded\n")
    except Exception as e:
        print(f"✗ Phase 7 failed: {e}")
        print("  Make sure you have accepted the Gemma license on Kaggle")
        return

    # ========================================================================
    # Phase 8: Convert checkpoint to NNX format
    # ========================================================================
    print("Phase 8: Converting checkpoint to Flax NNX format...")
    try:
        # Clean checkpoint directories
        os.system(f"rm -rf {INTERMEDIATE_CKPT_DIR}")
        os.system(f"rm -rf {CKPT_DIR}")

        if model_family == "gemma2":
            params = params_lib.load_and_format_params(
                os.path.join(kaggle_ckpt_path, "gemma2-2b-it")
            )
            gemma = gemma_lib.Transformer.from_params(params, version="2-2b-it")
            checkpointer = ocp.StandardCheckpointer()
            _, state = nnx.split(gemma)
            checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)
            checkpointer.wait_until_finished()

            # Delete intermediate model to save memory
            del params
            del gemma
            del state
            gc.collect()

        print("✓ Phase 8 complete: Checkpoint converted\n")
    except Exception as e:
        print(f"✗ Phase 8 failed: {e}")
        return

    # ========================================================================
    # Phase 9: Define model loading functions
    # ========================================================================
    print("Phase 9: Defining model loading functions...")

    def get_gemma_ref_model(ckpt_path):
        mesh = jax.make_mesh(*MESH)
        model_config = gemma_lib.ModelConfig.gemma2_2b()
        abs_gemma = nnx.eval_shape(
            lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
        )
        abs_state = nnx.state(abs_gemma)
        abs_state = jax.tree.map(
            lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
            abs_state,
            nnx.get_named_sharding(abs_state, mesh),
        )
        checkpointer = ocp.StandardCheckpointer()
        restored_params = checkpointer.restore(ckpt_path, target=abs_state)

        graph_def, _ = nnx.split(abs_gemma)
        gemma = nnx.merge(graph_def, restored_params)
        return gemma, mesh, model_config

    def get_lora_model(base_model, mesh):
        lora_provider = qwix.LoraProvider(
            module_path=(
                ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
                ".*attn_vec_einsum"
            ),
            rank=RANK,
            alpha=ALPHA,
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

    print("✓ Phase 9 complete: Model loading functions defined\n")

    # ========================================================================
    # Phase 10: Load reference model
    # ========================================================================
    print("Phase 10: Loading reference model (Gemma2-2b-it)...")
    try:
        if model_family == "gemma2":
            ref_model, mesh, model_config = get_gemma_ref_model(
                ckpt_path=os.path.join(INTERMEDIATE_CKPT_DIR, "state")
            )
        print("✓ Phase 10 complete: Reference model loaded\n")
    except Exception as e:
        print(f"✗ Phase 10 failed: {e}")
        return

    # ========================================================================
    # Phase 11: Apply LoRA to create policy model
    # ========================================================================
    print("Phase 11: Applying LoRA to create policy model...")
    try:
        lora_policy = get_lora_model(ref_model, mesh=mesh)
        print("  Policy model structure:")
        nnx.display(lora_policy)
        print("✓ Phase 11 complete: LoRA policy model created\n")
    except Exception as e:
        print(f"✗ Phase 11 failed: {e}")
        return

    # ========================================================================
    # Phase 12: Load tokenizer
    # ========================================================================
    print("Phase 12: Loading tokenizer...")
    try:
        if model_family == "gemma2":
            tokenizer = tokenizer_lib.Tokenizer(
                tokenizer_path=os.path.join(kaggle_ckpt_path, "tokenizer.model")
            )
        print("✓ Phase 12 complete: Tokenizer loaded\n")
    except Exception as e:
        print(f"✗ Phase 12 failed: {e}")
        return

    # ========================================================================
    # Phase 13: Define reward functions
    # ========================================================================
    print("Phase 13: Defining reward functions...")

    # RegEx for format matching
    match_format = re.compile(
        rf"^[\s]{{0,}}"
        rf"{reasoning_start}.+?{reasoning_end}.*?"
        rf"{solution_start}(.+?){solution_end}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )

    match_numbers = re.compile(
        rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
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
            response = completion
            score += 0.5 if response.count(reasoning_start) == 1 else -0.5
            score += 0.5 if response.count(reasoning_end) == 1 else -0.5
            score += 0.5 if response.count(solution_start) == 1 else -0.5
            score += 0.5 if response.count(solution_end) == 1 else -0.5
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

    def check_numbers(prompts, completions, answer, **kwargs):
        question = kwargs["question"]
        responses = completions

        extracted_responses = [
            guess.group(1) if (guess := match_numbers.search(r)) is not None else None
            for r in responses
        ]

        scores = []
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

    print("  - Reward functions: 4 defined")
    print("    * match_format_exactly (3 points)")
    print("    * match_format_approximately (±0.5 points per tag)")
    print("    * check_answer (3 points exact, partial credit)")
    print("    * check_numbers (1.5 points)")
    print("✓ Phase 13 complete: Reward functions defined\n")

    # ========================================================================
    # Phase 14: Define evaluation functions
    # ========================================================================
    print("Phase 14: Defining evaluation functions...")

    def generate(question, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None):
        """Given prompt, generates text."""
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
            max_generation_steps=768,
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
        """Computes accuracy and percentage of outputs matching the format."""
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

    print("✓ Phase 14 complete: Evaluation functions defined\n")

    # ========================================================================
    # Phase 15: Create sampler for evaluation
    # ========================================================================
    print("Phase 15: Creating sampler for pre-training evaluation...")
    try:
        sampler = sampler_lib.Sampler(
            transformer=lora_policy,
            tokenizer=tokenizer,
            cache_config=sampler_lib.CacheConfig(
                cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
                num_layers=model_config.num_layers,
                num_kv_heads=model_config.num_kv_heads,
                head_dim=model_config.head_dim,
            ),
        )
        print("✓ Phase 15 complete: Sampler created\n")
    except Exception as e:
        print(f"✗ Phase 15 failed: {e}")
        return

    # ========================================================================
    # Phase 16: Pre-training evaluation
    # ========================================================================
    print("Phase 16: Running pre-training evaluation on test set...")
    print("  (This may take a few minutes)")
    try:
        (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
            test_dataset,
            sampler,
            **GENERATION_CONFIGS["greedy"],
        )
        print(f"\n  Pre-training results:")
        print(f"    Correct: {corr}/{total}")
        print(f"    Accuracy: {accuracy:.2f}%")
        print(f"    Partial accuracy: {partial_accuracy:.2f}%")
        print(f"    Format accuracy: {format_accuracy:.2f}%")
        print("✓ Phase 16 complete: Pre-training evaluation done\n")
    except Exception as e:
        print(f"✗ Phase 16 failed: {e}")
        print("  Continuing with training setup...")

    # ========================================================================
    # Phase 17: Setup checkpointing and metrics logging
    # ========================================================================
    print("Phase 17: Setting up checkpointing and metrics logging...")
    try:
        checkpointing_options = ocp.CheckpointManagerOptions(
            save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
        )

        metrics_logging_options = metrics_logger.MetricsLoggerOptions(
            log_dir="/tmp/content/tmp/tensorboard/grpo", flush_every_n_steps=20
        )

        print(f"  Checkpoint dir: {CKPT_DIR}")
        print(f"  Save interval: every {SAVE_INTERVAL_STEPS} steps")
        print(f"  Max checkpoints to keep: {MAX_TO_KEEP}")
        print(f"  Tensorboard logs: /tmp/content/tmp/tensorboard/grpo")
        print("✓ Phase 17 complete: Checkpointing configured\n")
    except Exception as e:
        print(f"✗ Phase 17 failed: {e}")
        return

    # ========================================================================
    # Phase 18: Setup optimizer and learning rate schedule
    # ========================================================================
    print("Phase 18: Setting up optimizer and learning rate schedule...")
    try:
        optimizer = optax.adamw(
            learning_rate=optax.schedules.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=LEARNING_RATE,
                warmup_steps=WARMUP_STEPS,
                decay_steps=MAX_STEPS,
                end_value=0.0,
            ),
            b1=B1,
            b2=B2,
            weight_decay=WEIGHT_DECAY,
        )

        if MAX_GRAD_NORM is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
                optimizer,
            )

        print(f"  Optimizer: AdamW")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Warmup steps: {WARMUP_STEPS}")
        print(f"  Grad clipping: {MAX_GRAD_NORM}")
        print("✓ Phase 18 complete: Optimizer configured\n")
    except Exception as e:
        print(f"✗ Phase 18 failed: {e}")
        return

    # ========================================================================
    # Phase 19: Create RL cluster configuration
    # ========================================================================
    print("Phase 19: Creating RL cluster configuration...")
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
                eval_every_n_steps=EVAL_EVERY_N_STEPS,
                max_steps=MAX_STEPS,
                mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
                train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
                metrics_logging_options=metrics_logging_options,
                checkpoint_root_directory=CKPT_DIR,
                checkpointing_options=checkpointing_options,
            ),
            rollout_config=base_rollout.RolloutConfig(
                max_tokens_to_generate=TOTAL_GENERATION_STEPS,
                max_prompt_length=MAX_PROMPT_LENGTH,
                kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
            ),
        )

        grpo_config = GRPOConfig(
            num_generations=NUM_GENERATIONS,
            num_iterations=NUM_ITERATIONS,
            beta=BETA,
            epsilon=EPSILON,
        )

        print("  RL Cluster configured with:")
        print(f"    - Actor, Reference, and Rollout roles")
        print(f"    - Rollout engine: vanilla")
        print(f"    - Max training steps: {MAX_STEPS}")
        print("✓ Phase 19 complete: RL cluster configured\n")
    except Exception as e:
        print(f"✗ Phase 19 failed: {e}")
        return

    # ========================================================================
    # Phase 20: Initialize RL cluster and GRPO learner
    # ========================================================================
    print("Phase 20: Initializing RL cluster and GRPO learner...")
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
        print("✓ Phase 20 complete: GRPO trainer ready\n")
    except Exception as e:
        print(f"✗ Phase 20 failed: {e}")
        return

    # ========================================================================
    # Phase 21: Run GRPO training
    # ========================================================================
    print("Phase 21: Starting GRPO training...")
    print("  Note: First training step may take up to 5 minutes")
    print("  This is a long-running process. Press Ctrl+C to stop.")
    print()

    try:
        with mesh:
            grpo_trainer.train(train_dataset)
        print("\n✓ Phase 21 complete: Training finished\n")
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user\n")
    except Exception as e:
        print(f"\n✗ Phase 21 failed: {e}")
        print("  Training encountered an error")
        return

    # ========================================================================
    # Phase 22: Load trained checkpoint
    # ========================================================================
    print("Phase 22: Loading trained checkpoint...")
    try:
        trained_ckpt_path = os.path.join(
            CKPT_DIR, "actor", str(MAX_STEPS), "model_params"
        )

        abs_params = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            nnx.state(lora_policy, nnx.LoRAParam),
        )
        checkpointer = ocp.StandardCheckpointer()
        trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

        nnx.update(
            lora_policy,
            jax.tree.map(
                lambda a, b: b,
                nnx.state(lora_policy, nnx.LoRAParam),
                trained_lora_params,
            ),
        )

        print(f"  Loaded checkpoint from: {trained_ckpt_path}")
        print("✓ Phase 22 complete: Trained model loaded\n")
    except Exception as e:
        print(f"✗ Phase 22 failed: {e}")
        print("  Could not load trained checkpoint")
        return

    # ========================================================================
    # Phase 23: Post-training evaluation
    # ========================================================================
    print("Phase 23: Running post-training evaluation on test set...")
    print("  (This may take a few minutes)")
    try:
        sampler = sampler_lib.Sampler(
            transformer=lora_policy,
            tokenizer=tokenizer,
            cache_config=sampler_lib.CacheConfig(
                cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
                num_layers=model_config.num_layers,
                num_kv_heads=model_config.num_kv_heads,
                head_dim=model_config.head_dim,
            ),
        )

        (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
            test_dataset,
            sampler,
            **GENERATION_CONFIGS["greedy"],
        )

        print(f"\n  Post-training results:")
        print(f"    Correct: {corr}/{total}")
        print(f"    Accuracy: {accuracy:.2f}%")
        print(f"    Partial accuracy: {partial_accuracy:.2f}%")
        print(f"    Format accuracy: {format_accuracy:.2f}%")
        print("✓ Phase 23 complete: Post-training evaluation done\n")
    except Exception as e:
        print(f"✗ Phase 23 failed: {e}")

    # ========================================================================
    # Complete!
    # ========================================================================
    print("=" * 60)
    print("GRPO Training Complete!")
    print("=" * 60)
    print()
    print("Checkpoints saved to:", CKPT_DIR)
    print("TensorBoard logs:", "/tmp/content/tmp/tensorboard/grpo")
    print()


if __name__ == "__main__":
    main()
