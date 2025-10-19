# Mousse

A simple LLM chat interface using MaxText and JAX.

## Installation

```bash
# Create virtual environment
$ uv venv ~/.venvs/mousse_env

# Activate virtual environment
$ source ~/.venvs/mousse_env/bin/activate

# Install mousse (this will also install maxtext as a dependency)
$ uv pip install -e .
```

## Getting a Model Checkpoint

You need a Gemma 2B checkpoint converted to MaxText format. Options:

1. **Download from HuggingFace and convert:**
   ```bash
   # Download Gemma 2B from HuggingFace (requires authentication)
   # Then convert using MaxText's conversion script
   ```

2. **Set checkpoint path:**
   ```bash
   $ export MOUSSE_CHECKPOINT_PATH=/path/to/gemma2-2b/checkpoint
   ```

See [MaxText documentation](https://github.com/AI-Hypercomputer/maxtext) for checkpoint conversion details.

## Usage

Set the checkpoint path:
```bash
$ export MOUSSE_CHECKPOINT_PATH=/path/to/checkpoint
```

### Interactive mode:
```bash
$ python -m mousse
```

### Single prompt:
```bash
$ python -m mousse "What is the capital of France?"
```

### With custom config:
```bash
$ python -m mousse --config=/path/to/config.yml
```

## Configuration

Mousse uses MaxText's configuration system. Key settings:
- `MOUSSE_CHECKPOINT_PATH` (env var): Path to model checkpoint
- `model_name`: Model architecture (default: gemma2-2b)
- Pass additional MaxText config via command line or config file

See MaxText documentation for full configuration options.

## About

Mousse is powered by:
- **MaxText**: High-performance LLM library in pure Python/JAX
- **JAX**: High-performance numerical computing
- **No PyTorch**: Pure JAX implementation
