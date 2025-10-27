"""
Model brand configuration for chic
Maps model names to their Kaggle paths and families
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import jax


class ModelFamily(Enum):
    """Model family enumeration with Kaggle paths."""
    GEMMA = ('gemma', 'Google/gemma/flax/')
    GEMMA2 = ('gemma2', 'Google/gemma-2/flax/')
    GEMMA3 = ('gemma3', 'Google/gemma-3/flax/')

    def __init__(self, family_name: str, kaggle_path: str):
        self.family_name = family_name
        self.kaggle_path = kaggle_path


@dataclass(frozen=True)
class ModelBrand:
    """Configuration for a specific model variant."""
    name: str
    model_family: ModelFamily

    @property
    def full_kaggle_path(self) -> str:
        """Get the full Kaggle download path."""
        return f"{self.model_family.kaggle_path}{self.name}"

    @staticmethod
    def get_by_name(model_name: str) -> ModelBrand:
        """Get ModelBrand for a given model name."""
        if model_name not in MODEL_BY_NAME:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Supported models: {', '.join(MODEL_NAMES)}"
            )
        return MODEL_BY_NAME[model_name]

    def get_model_config(self) -> Any:
        """Get the model config object for this model."""
        if self.model_family == ModelFamily.GEMMA3:
            from tunix.models.gemma3 import model as gemma3_lib
            # Map all gemma3 variants to their base size configs
            config_map = {
                'gemma-3-270m': gemma3_lib.ModelConfig.gemma3_270m,
                'gemma-3-270m-it': gemma3_lib.ModelConfig.gemma3_270m,
                'gemma3-1b': gemma3_lib.ModelConfig.gemma3_1b,
                'gemma3-1b-it': gemma3_lib.ModelConfig.gemma3_1b,
                'gemma3-4b': gemma3_lib.ModelConfig.gemma3_4b,
                'gemma3-4b-it': gemma3_lib.ModelConfig.gemma3_4b,
                'gemma3-12b': gemma3_lib.ModelConfig.gemma3_12b,
                'gemma3-12b-it': gemma3_lib.ModelConfig.gemma3_12b,
                'gemma3-27b': gemma3_lib.ModelConfig.gemma3_27b,
                'gemma3-27b-it': gemma3_lib.ModelConfig.gemma3_27b,
            }
            return config_map[self.name]()
        elif self.model_family == ModelFamily.GEMMA2:
            from tunix.models.gemma import model as gemma_lib
            config_map = {
                'gemma2-2b': gemma_lib.ModelConfig.gemma2_2b,
                'gemma2-2b-it': gemma_lib.ModelConfig.gemma2_2b,
                'gemma2-9b': gemma_lib.ModelConfig.gemma2_9b,
                'gemma2-9b-it': gemma_lib.ModelConfig.gemma2_9b,
                # Note: gemma2-27b not yet supported in tunix
            }
            if self.name not in config_map:
                raise ValueError(f"Model {self.name} not yet supported in tunix library")
            return config_map[self.name]()
        elif self.model_family == ModelFamily.GEMMA:
            from tunix.models.gemma import model as gemma_lib
            config_map = {
                'gemma-2b': gemma_lib.ModelConfig.gemma_2b,
                'gemma-2b-it': gemma_lib.ModelConfig.gemma_2b,
                'gemma-1.1-2b-it': gemma_lib.ModelConfig.gemma_2b,
                'gemma-7b': gemma_lib.ModelConfig.gemma_7b,
                'gemma-7b-it': gemma_lib.ModelConfig.gemma_7b,
                'gemma-1.1-7b-it': gemma_lib.ModelConfig.gemma_7b,
            }
            return config_map[self.name]()
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

    def load_model(self, ckpt_path: str, mesh_config: tuple) -> tuple[Any, Any, Any]:
        """Load model from checkpoint.

        Returns:
            tuple: (model, mesh, model_config)
        """
        import functools
        import orbax.checkpoint as ocp
        from flax import nnx
        from jax import numpy as jnp

        mesh = jax.make_mesh(*mesh_config)
        model_config = self.get_model_config()
        checkpoint_path = os.path.join(ckpt_path, self.name)

        # Use the correct loader based on model family
        if self.model_family == ModelFamily.GEMMA3:
            from tunix.models.gemma3 import params as params_module
            model = params_module.create_model_from_checkpoint(
                checkpoint_path, model_config, mesh
            )
        elif self.model_family in (ModelFamily.GEMMA2, ModelFamily.GEMMA):
            # Load Gemma/Gemma2 models (which don't have create_model_from_checkpoint)
            from tunix.models.gemma import model as gemma_model_lib
            from tunix.models.gemma import params as gemma_params

            # Create model structure
            abs_model = nnx.eval_shape(
                lambda: gemma_model_lib.Transformer(model_config, rngs=nnx.Rngs(0))
            )

            # Load params from checkpoint
            params = gemma_params.load_and_format_params(checkpoint_path)

            # Apply sharding if mesh provided
            dtype = jnp.bfloat16
            if mesh is not None:
                params = jax.tree.map(
                    lambda x, shd: jnp.asarray(x, device=shd, dtype=dtype),
                    params,
                    nnx.to_pure_dict(nnx.get_named_sharding(nnx.state(abs_model), mesh)),
                )
            else:
                params = jax.tree.map(functools.partial(jnp.asarray, dtype=dtype), params)

            nnx.update(abs_model, params)
            model = abs_model
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

        return model, mesh, model_config

    def get_tokenizer_path(self, kaggle_ckpt_path: str) -> str:
        """Get the tokenizer path for this model."""
        return os.path.join(kaggle_ckpt_path, "tokenizer.model")


# Define all supported models
_MODELS = [
    # Gemma (v1) - ordered by size
    ModelBrand('gemma-2b', ModelFamily.GEMMA),
    ModelBrand('gemma-2b-it', ModelFamily.GEMMA),
    ModelBrand('gemma-1.1-2b-it', ModelFamily.GEMMA),
    ModelBrand('gemma-7b', ModelFamily.GEMMA),
    ModelBrand('gemma-7b-it', ModelFamily.GEMMA),
    ModelBrand('gemma-1.1-7b-it', ModelFamily.GEMMA),

    # Gemma2 - ordered by size
    ModelBrand('gemma2-2b', ModelFamily.GEMMA2),
    ModelBrand('gemma2-2b-it', ModelFamily.GEMMA2),
    ModelBrand('gemma2-9b', ModelFamily.GEMMA2),
    ModelBrand('gemma2-9b-it', ModelFamily.GEMMA2),
    ModelBrand('gemma2-27b', ModelFamily.GEMMA2),
    ModelBrand('gemma2-27b-it', ModelFamily.GEMMA2),

    # Gemma3 - ordered by size
    ModelBrand('gemma-3-270m', ModelFamily.GEMMA3),
    ModelBrand('gemma-3-270m-it', ModelFamily.GEMMA3),
    ModelBrand('gemma3-1b', ModelFamily.GEMMA3),
    ModelBrand('gemma3-1b-it', ModelFamily.GEMMA3),
    ModelBrand('gemma3-4b', ModelFamily.GEMMA3),
    ModelBrand('gemma3-4b-it', ModelFamily.GEMMA3),
    ModelBrand('gemma3-12b', ModelFamily.GEMMA3),
    ModelBrand('gemma3-12b-it', ModelFamily.GEMMA3),
    ModelBrand('gemma3-27b', ModelFamily.GEMMA3),
    ModelBrand('gemma3-27b-it', ModelFamily.GEMMA3),
]


# Create lookup dictionaries
MODEL_BY_NAME: dict[str, ModelBrand] = {model.name: model for model in _MODELS}
MODEL_NAMES: list[str] = [model.name for model in _MODELS]
