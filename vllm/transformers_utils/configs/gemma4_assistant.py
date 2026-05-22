# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 assistant draft model configuration.

This local config lets vLLM load Gemma4 MTP assistant checkpoints when the
installed Transformers version does not yet expose Gemma4AssistantConfig.
"""

from transformers import PretrainedConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig


class Gemma4AssistantConfig(PretrainedConfig):
    model_type = "gemma4_assistant"
    sub_configs = {"text_config": Gemma4TextConfig}

    def __init__(
        self,
        text_config=None,
        backbone_hidden_size: int = 0,
        tie_word_embeddings: bool = True,
        use_ordered_embeddings: bool = False,
        num_centroids: int = 2048,
        centroid_intermediate_top_k: int = 32,
        **kwargs,
    ) -> None:
        if isinstance(text_config, dict):
            self.text_config = Gemma4TextConfig(**text_config)
        elif text_config is None:
            self.text_config = Gemma4TextConfig()
        else:
            self.text_config = text_config

        self.backbone_hidden_size = backbone_hidden_size
        self.use_ordered_embeddings = use_ordered_embeddings
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        super().__init__(**kwargs)
        self.tie_word_embeddings = tie_word_embeddings

    def get_text_config(self, *args, **kwargs):
        return self.text_config


__all__ = ["Gemma4AssistantConfig"]
