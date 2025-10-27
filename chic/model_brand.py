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
    """Model family enumeration."""
    GEMMA = 'gemma'
    GEMMA2 = 'gemma2'
    GEMMA3 = 'gemma3'
    LLAMA3 = 'llama3'
    QWEN2 = 'qwen2'
    QWEN3 = 'qwen3'


@dataclass(frozen=True)
class ModelBrand:
    """Configuration for a specific model variant."""
    name: str
    model_family: ModelFamily
    kaggle_path: str

    @property
    def full_kaggle_path(self) -> str:
        """Get the full Kaggle download path."""
        return f"{self.kaggle_path}{self.name}"

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
        # elif self.model_family == ModelFamily.LLAMA3:
        #     from tunix.models.llama3 import model as llama3_lib
        #     config_map = {
        #         'llama3.2-1b-instruct': llama3_lib.ModelConfig.llama3_2_1b,
        #         'llama3.2-3b-instruct': llama3_lib.ModelConfig.llama3_2_3b,
        #         'llama3.1-8b-instruct': llama3_lib.ModelConfig.llama3_1_8b,
        #         'llama3-70b-instruct': llama3_lib.ModelConfig.llama3_70b,
        #         'llama3-405b-instruct': llama3_lib.ModelConfig.llama3_405b,
        #     }
        #     return config_map[self.name]()
        # elif self.model_family == ModelFamily.QWEN2:
        #     from tunix.models.qwen2 import model as qwen2_lib
        #     config_map = {
        #         'qwen2.5-0.5b-instruct': qwen2_lib.ModelConfig.qwen2_5_0_5b,
        #         'deepseek-r1-distill-qwen-1.5b': qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1_5b,
        #         'qwen2.5-1.5b-instruct': qwen2_lib.ModelConfig.qwen2_5_1_5b,
        #         'qwen2.5-3b-instruct': qwen2_lib.ModelConfig.qwen2_5_3b,
        #         'qwen2.5-7b-instruct': qwen2_lib.ModelConfig.qwen2_5_7b,
        #     }
        #     return config_map[self.name]()
        # elif self.model_family == ModelFamily.QWEN3:
        #     from tunix.models.qwen3 import model as qwen3_lib
        #     config_map = {
        #         'qwen3-0.6b-instruct': qwen3_lib.ModelConfig.qwen3_0_6b,
        #         'qwen3-1.7b-instruct': qwen3_lib.ModelConfig.qwen3_1_7b,
        #         'qwen3-8b-instruct': qwen3_lib.ModelConfig.qwen3_8b,
        #         'qwen3-14b-instruct': qwen3_lib.ModelConfig.qwen3_14b,
        #         'qwen3-30b-instruct': qwen3_lib.ModelConfig.qwen3_30b,
        #     }
        #     return config_map[self.name]()
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
        # elif self.model_family == ModelFamily.LLAMA3:
        #     from tunix.models.llama3 import params as llama3_params
        #     checkpoint_path = os.path.join(ckpt_path, self.name)
        #     model = llama3_params.create_model_from_checkpoint(
        #         checkpoint_path, model_config, mesh
        #     )
        # elif self.model_family in (ModelFamily.QWEN2, ModelFamily.QWEN3):
        #     from tunix.models.qwen2 import params as qwen2_params
        #     checkpoint_path = os.path.join(ckpt_path, self.name)
        #     model = qwen2_params.create_model_from_checkpoint(
        #         checkpoint_path, model_config, mesh
        #     )
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

        return model, mesh, model_config

    def get_tokenizer_path(self, kaggle_ckpt_path: str) -> str:
        """Get the tokenizer path for this model."""
        return os.path.join(kaggle_ckpt_path, "tokenizer.model")


# Define all supported models
_MODELS = [
    # Gemma (v1) - ordered by size
    ModelBrand('gemma-2b', ModelFamily.GEMMA, 'Google/gemma/flax/'),
    ModelBrand('gemma-2b-it', ModelFamily.GEMMA, 'Google/gemma/flax/'),
    ModelBrand('gemma-1.1-2b-it', ModelFamily.GEMMA, 'Google/gemma/flax/'),
    ModelBrand('gemma-7b', ModelFamily.GEMMA, 'Google/gemma/flax/'),
    ModelBrand('gemma-7b-it', ModelFamily.GEMMA, 'Google/gemma/flax/'),
    ModelBrand('gemma-1.1-7b-it', ModelFamily.GEMMA, 'Google/gemma/flax/'),

    # Gemma2 - ordered by size
    ModelBrand('gemma2-2b', ModelFamily.GEMMA2, 'Google/gemma-2/flax/'),
    ModelBrand('gemma2-2b-it', ModelFamily.GEMMA2, 'Google/gemma-2/flax/'),
    ModelBrand('gemma2-9b', ModelFamily.GEMMA2, 'Google/gemma-2/flax/'),
    ModelBrand('gemma2-9b-it', ModelFamily.GEMMA2, 'Google/gemma-2/flax/'),
    ModelBrand('gemma2-27b', ModelFamily.GEMMA2, 'Google/gemma-2/flax/'),
    ModelBrand('gemma2-27b-it', ModelFamily.GEMMA2, 'Google/gemma-2/flax/'),

    # Gemma3 - ordered by size
    ModelBrand('gemma-3-270m', ModelFamily.GEMMA3, 'Google/gemma-3/flax/'),
    ModelBrand('gemma-3-270m-it', ModelFamily.GEMMA3, 'Google/gemma-3/flax/'),
    ModelBrand('gemma3-1b', ModelFamily.GEMMA3, 'Google/gemma-3/flax/'),
    ModelBrand('gemma3-1b-it', ModelFamily.GEMMA3, 'Google/gemma-3/flax/'),
    ModelBrand('gemma3-4b', ModelFamily.GEMMA3, 'Google/gemma-3/flax/'),
    ModelBrand('gemma3-4b-it', ModelFamily.GEMMA3, 'Google/gemma-3/flax/'),
    ModelBrand('gemma3-12b', ModelFamily.GEMMA3, 'Google/gemma-3/flax/'),
    ModelBrand('gemma3-12b-it', ModelFamily.GEMMA3, 'Google/gemma-3/flax/'),
    ModelBrand('gemma3-27b', ModelFamily.GEMMA3, 'Google/gemma-3/flax/'),
    ModelBrand('gemma3-27b-it', ModelFamily.GEMMA3, 'Google/gemma-3/flax/'),

    # # Llama3 - TODO: Verify Flax availability on Kaggle
    # ModelBrand('llama3.2-1b-instruct', ModelFamily.LLAMA3, 'meta-llama/llama-3/flax/'),
    # ModelBrand('llama3.2-3b-instruct', ModelFamily.LLAMA3, 'meta-llama/llama-3/flax/'),
    # ModelBrand('llama3.1-8b-instruct', ModelFamily.LLAMA3, 'meta-llama/llama-3/flax/'),
    # ModelBrand('llama3-70b-instruct', ModelFamily.LLAMA3, 'meta-llama/llama-3/flax/'),
    # ModelBrand('llama3-405b-instruct', ModelFamily.LLAMA3, 'meta-llama/llama-3/flax/'),

    # # Qwen2 - TODO: Verify Flax availability on Kaggle
    # ModelBrand('qwen2.5-0.5b-instruct', ModelFamily.QWEN2, 'Qwen/Qwen2.5/flax/'),
    # ModelBrand('deepseek-r1-distill-qwen-1.5b', ModelFamily.QWEN2, 'deepseek-ai/DeepSeek-R1-Distill-Qwen/flax/'),
    # ModelBrand('qwen2.5-1.5b-instruct', ModelFamily.QWEN2, 'Qwen/Qwen2.5/flax/'),
    # ModelBrand('qwen2.5-3b-instruct', ModelFamily.QWEN2, 'Qwen/Qwen2.5/flax/'),
    # ModelBrand('qwen2.5-7b-instruct', ModelFamily.QWEN2, 'Qwen/Qwen2.5/flax/'),

    # # Qwen3 - TODO: Verify Flax availability on Kaggle
    # ModelBrand('qwen3-0.6b-instruct', ModelFamily.QWEN3, 'Qwen/Qwen3/flax/'),
    # ModelBrand('qwen3-1.7b-instruct', ModelFamily.QWEN3, 'Qwen/Qwen3/flax/'),
    # ModelBrand('qwen3-8b-instruct', ModelFamily.QWEN3, 'Qwen/Qwen3/flax/'),
    # ModelBrand('qwen3-14b-instruct', ModelFamily.QWEN3, 'Qwen/Qwen3/flax/'),
    # ModelBrand('qwen3-30b-instruct', ModelFamily.QWEN3, 'Qwen/Qwen3/flax/'),
]


# Create lookup dictionaries
MODEL_BY_NAME: dict[str, ModelBrand] = {model.name: model for model in _MODELS}
MODEL_NAMES: list[str] = [model.name for model in _MODELS]
