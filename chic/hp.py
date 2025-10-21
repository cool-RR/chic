#!/usr/bin/env python
"""
Hyperparameter search for GRPO training
Calls the main training script with different hyperparameter combinations
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click

from chic.json_tools import JsonlaReader, JsonlaWriter
from chic.trekking import CHIC_HOME, Trek


# ANSI color codes for terminal formatting
class Color:
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'


def generate_hp_config(trial_num: int, base_config: dict) -> dict:
    """Generate a hyperparameter configuration for a trial."""
    # Define search spaces for key hyperparameters
    # Memory-aware: reduced lora_rank/alpha, fixed num_generations
    hp_spaces = {
        'learning_rate': [1e-6, 3e-6, 5e-6, 1e-5, 3e-5],
        'beta': [0.01, 0.05, 0.08, 0.1, 0.15, 0.2],
        'epsilon': [0.1, 0.2, 0.3, 0.4],
        'temperature': [0.7, 0.8, 0.9, 1.0, 1.1],
        'lora_rank': [32, 64, 128],
        'lora_alpha': [32, 64, 128],
        'num_iterations': [1, 2, 3, 5],
    }

    config = base_config.copy()

    # Randomly sample from each hyperparameter space
    for hp_name, hp_values in hp_spaces.items():
        config[hp_name] = random.choice(hp_values)

    # Keep num_generations fixed at 2 for memory constraints
    config['num_generations'] = 2

    # Ensure lora_alpha >= lora_rank for stability
    if config['lora_alpha'] < config['lora_rank']:
        config['lora_alpha'] = config['lora_rank']

    return config


def get_latest_trek_folder() -> Path:
    """Get the most recently created Trek folder."""
    trek_folders = [f for f in CHIC_HOME.iterdir() if f.is_dir()]
    if not trek_folders:
        raise ValueError(f'No Trek folders found in {CHIC_HOME}')
    return max(trek_folders)


def parse_trek_results(trek_folder: Path) -> dict | None:
    """Parse results from Trek's jsonla files."""
    results_path = trek_folder / 'results.jsonla'

    if not results_path.exists():
        return None

    try:
        reader = JsonlaReader(results_path)
        results = list(reader)

        if not results:
            return None

        # Find pre-training and post-training results
        pre_train = None
        post_train_results = []

        for row in results:
            if row.get('phase') == 'pre_training':
                pre_train = row
            elif row.get('phase') == 'post_training':
                post_train_results.append(row)

        if pre_train is None or not post_train_results:
            return None

        # Get final post-training result (last iteration)
        post_train = post_train_results[-1]

        return {
            'pre_train_accuracy': pre_train['accuracy'],
            'pre_train_partial_accuracy': pre_train.get('partial_accuracy', 0.0),
            'pre_train_format_accuracy': pre_train.get('format_accuracy', 0.0),
            'post_train_accuracy': post_train['accuracy'],
            'post_train_partial_accuracy': post_train.get('partial_accuracy', 0.0),
            'post_train_format_accuracy': post_train.get('format_accuracy', 0.0),
            'improvement': post_train['accuracy'] - pre_train['accuracy'],
        }
    except Exception as e:
        print(f"Error reading Trek results: {e}")
        return None


@click.command()
@click.option('--num-trials', default=2, show_default=True,
              help='Number of hyperparameter combinations to try')
@click.option('--train-script', default='chic/lol.py', show_default=True,
              help='Path to training script')
# Pass-through options for the training script
@click.option('--train-data-dir', default='./data/train', show_default=True)
@click.option('--test-data-dir', default='./data/test', show_default=True)
@click.option('--train-fraction', default=1.0, show_default=True)
@click.option('--data-source', type=click.Choice(['tfds', 'kaggle']), default='kaggle',
              show_default=True)
@click.option('--max-prompt-length', default=128, show_default=True)
@click.option('--total-generation-steps', default=256, show_default=True)
@click.option('--top-p', default=1.0, show_default=True)
@click.option('--top-k', default=50, show_default=True)
@click.option('--num-iterations', default=1, show_default=True)
@click.option('--train-micro-batch-size', default=1, show_default=True)
@click.option('--num-batches', default=50, show_default=True)
@click.option('--num-test-batches', default=30, show_default=True)
@click.option('--eval-every-n-steps', default=10, show_default=True)
@click.option('--num-epochs', default=1, show_default=True)
@click.option('--b1', default=0.9, show_default=True)
@click.option('--b2', default=0.99, show_default=True)
@click.option('--weight-decay', default=0.1, show_default=True)
@click.option('--max-grad-norm', default=0.1, show_default=True)
@click.option('--save-interval-steps', default=500, show_default=True)
@click.option('--max-to-keep', default=4, show_default=True)
@click.option('--model-family', type=click.Choice(['gemma3']), default='gemma3', show_default=True)
@click.option('--model-version', default='gemma3-1b-it', show_default=True)
@click.option('--offload-to-cpu/--no-offload-to-cpu', default=False, show_default=True)
def main(num_trials, train_script, **kwargs):
    """Run hyperparameter search for GRPO training."""

    print("=" * 80)
    print(f"{Color.BOLD}{Color.CYAN}HYPERPARAMETER SEARCH{Color.END}")
    print("=" * 80)
    print(f"Number of trials: {num_trials}")
    print()

    # Create Trek for this hyperparameter search
    trek = Trek()
    comparison_writer = JsonlaWriter(trek.folder / 'comparison.jsonla')

    print(f"{Color.CYAN}Trek folder: {trek.folder}{Color.END}")
    print()

    # Store all results
    all_results = []

    # Base configuration from command-line args (for hyperparameters we'll vary)
    base_config = {
        'learning_rate': 3e-6,
        'beta': 0.08,
        'epsilon': 0.2,
        'temperature': 0.9,
        'lora_rank': 64,
        'lora_alpha': 64.0,
        'num_generations': 2,  # Fixed for memory constraints
        'num_iterations': 1,
    }

    # Run trials
    for trial_num in range(num_trials):
        print(f"\n{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}")
        print(f"{Color.BOLD}{Color.CYAN}TRIAL {trial_num + 1}/{num_trials}{Color.END}")
        print(f"{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}\n")

        # Generate hyperparameter configuration for this trial
        hp_config = generate_hp_config(trial_num, base_config)

        print(f"{Color.YELLOW}Hyperparameters for this trial:{Color.END}")
        for key, value in hp_config.items():
            print(f"  {key}: {value}")
        print()

        # Build command to run training script
        cmd = [
            sys.executable,
            train_script,
            f"--learning-rate={hp_config['learning_rate']}",
            f"--beta={hp_config['beta']}",
            f"--epsilon={hp_config['epsilon']}",
            f"--temperature={hp_config['temperature']}",
            f"--lora-rank={hp_config['lora_rank']}",
            f"--lora-alpha={hp_config['lora_alpha']}",
            f"--num-generations={hp_config['num_generations']}",
            f"--num-iterations={hp_config['num_iterations']}",
            "--dont-show-conversation",  # Always disable conversation display for HP search
        ]

        # Add all other options from kwargs
        for key, value in kwargs.items():
            key_name = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key_name}")
            else:
                cmd.append(f"--{key_name}={value}")

        start_time = datetime.now()

        print(f"{Color.YELLOW}Running training...{Color.END}")

        try:
            # Run the training script silently (Trek will handle logging)
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3600  # 1 hour timeout per trial
            )

            end_time = datetime.now()
            duration = end_time - start_time

            # Get the Trek folder that was just created
            trek_folder = get_latest_trek_folder()

            # Parse metrics from Trek's jsonla files
            metrics = parse_trek_results(trek_folder)

            if metrics is None or result.returncode != 0:
                raise Exception(f"Training failed with return code {result.returncode}. "
                                f"Trek folder: {trek_folder}")

            # Store results
            trial_result = {
                'trial_num': trial_num + 1,
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'trek_folder': str(trek_folder),
                'status': 'success',
                'error_message': None,
                # Hyperparameters
                'learning_rate': hp_config['learning_rate'],
                'beta': hp_config['beta'],
                'epsilon': hp_config['epsilon'],
                'temperature': hp_config['temperature'],
                'lora_rank': hp_config['lora_rank'],
                'lora_alpha': hp_config['lora_alpha'],
                'num_generations': hp_config['num_generations'],
                'num_iterations': hp_config['num_iterations'],
                # Results
                'pre_train_accuracy': metrics['pre_train_accuracy'],
                'post_train_accuracy': metrics['post_train_accuracy'],
                'improvement': metrics['improvement'],
                'pre_train_partial_accuracy': metrics['pre_train_partial_accuracy'],
                'post_train_partial_accuracy': metrics['post_train_partial_accuracy'],
                'pre_train_format_accuracy': metrics['pre_train_format_accuracy'],
                'post_train_format_accuracy': metrics['post_train_format_accuracy'],
            }
            all_results.append(trial_result)

            # Write to comparison.jsonla immediately
            comparison_writer.write(trial_result)

            print(f"\n{Color.GREEN}✓ Trial {trial_num + 1} complete{Color.END}")
            print(f"  Duration: {duration}")
            print(f"  Improvement: {metrics['improvement']:.2f}%")
            print(f"  Trek folder: {trek_folder}")

        except subprocess.TimeoutExpired:
            print(f"\n{Color.RED}✗ Trial {trial_num + 1} timed out{Color.END}")
            trial_result = {
                'trial_num': trial_num + 1,
                'timestamp': start_time.isoformat(),
                'duration_seconds': None,
                'trek_folder': None,
                'status': 'timeout',
                'error_message': 'Training exceeded timeout limit',
                # Hyperparameters
                'learning_rate': hp_config['learning_rate'],
                'beta': hp_config['beta'],
                'epsilon': hp_config['epsilon'],
                'temperature': hp_config['temperature'],
                'lora_rank': hp_config['lora_rank'],
                'lora_alpha': hp_config['lora_alpha'],
                'num_generations': hp_config['num_generations'],
                'num_iterations': hp_config['num_iterations'],
                # Results (all None)
                'pre_train_accuracy': None,
                'post_train_accuracy': None,
                'improvement': None,
                'pre_train_partial_accuracy': None,
                'post_train_partial_accuracy': None,
                'pre_train_format_accuracy': None,
                'post_train_format_accuracy': None,
            }
            comparison_writer.write(trial_result)

        except Exception as e:
            print(f"\n{Color.RED}✗ Trial {trial_num + 1} failed: {e}{Color.END}")
            trial_result = {
                'trial_num': trial_num + 1,
                'timestamp': start_time.isoformat(),
                'duration_seconds': None,
                'trek_folder': None,
                'status': 'failed',
                'error_message': str(e),
                # Hyperparameters
                'learning_rate': hp_config['learning_rate'],
                'beta': hp_config['beta'],
                'epsilon': hp_config['epsilon'],
                'temperature': hp_config['temperature'],
                'lora_rank': hp_config['lora_rank'],
                'lora_alpha': hp_config['lora_alpha'],
                'num_generations': hp_config['num_generations'],
                'num_iterations': hp_config['num_iterations'],
                # Results (all None)
                'pre_train_accuracy': None,
                'post_train_accuracy': None,
                'improvement': None,
                'pre_train_partial_accuracy': None,
                'post_train_partial_accuracy': None,
                'pre_train_format_accuracy': None,
                'post_train_format_accuracy': None,
            }
            comparison_writer.write(trial_result)

    # Sort results by improvement
    successful_results = [r for r in all_results if r['status'] == 'success']
    successful_results.sort(key=lambda x: x['improvement'], reverse=True)

    # Display top 5 performers
    print(f"\n\n{Color.BOLD}{Color.GREEN}{'='*80}{Color.END}")
    print(f"{Color.BOLD}{Color.GREEN}TOP 5 PERFORMERS{Color.END}")
    print(f"{Color.BOLD}{Color.GREEN}{'='*80}{Color.END}\n")

    top_n = min(5, len(successful_results))
    for i, result in enumerate(successful_results[:top_n]):
        print(f"{Color.BOLD}#{i+1} - Trial {result['trial_num']}{Color.END}")
        print(f"  Improvement: {Color.GREEN}{result['improvement']:.2f}%{Color.END}")
        print(f"  Post-training accuracy: {result['post_train_accuracy']:.2f}%")
        print(f"  Duration: {result['duration_seconds']:.1f}s")
        print(f"  Trek folder: {result['trek_folder']}")
        print(f"  Hyperparameters:")
        print(f"    learning_rate: {result['learning_rate']}")
        print(f"    beta: {result['beta']}")
        print(f"    epsilon: {result['epsilon']}")
        print(f"    temperature: {result['temperature']}")
        print(f"    lora_rank: {result['lora_rank']}")
        print(f"    lora_alpha: {result['lora_alpha']}")
        print()

    print(f"\n{Color.CYAN}Hyperparameter search complete!{Color.END}")
    print(f"{Color.CYAN}Trek folder: {trek.folder.as_posh()}{Color.END}")
    print(f"{Color.CYAN}Comparison file: {(trek.folder / 'comparison.jsonla').as_posh()}{Color.END}\n")


if __name__ == "__main__":
    main()
