"""Real tiny CPU Qwen inference without network access or downloaded weights."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from target.model import Model
from target.qwen_model import QwenModel


def model(directory: Path, *, tokens: int, context: int, repetition: float = 1.0) -> Model:
    """Assemble the production adapter with a real tokenizer and tiny native model.

    Args:
        directory (Path): Temporary tokenizer resource directory.
        tokens (int): Normal generation allowance.
        context (int): Native model context capacity.
        repetition (float, default=1.0): Inherited native repetition penalty.

    Returns:
        Model: Offline production Qwen model with no inference mocks.
    """
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "added_tokens": [],
                "model": {
                    "type": "WordLevel",
                    "vocab": {
                        "<unk>": 0,
                        "<eos>": 1,
                        "word": 2,
                        "reply": 3
                    },
                    "unk_token": "<unk>"
                },
                "pre_tokenizer": {"type": "Whitespace"}
            }
        ),
        encoding="utf-8"
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "model_input_names": [
                    "input_ids", "attention_mask"
                ],
                "chat_template": "{% for message in messages %}{{ message['content'] }} {% endfor %}{% if add_generation_prompt %}reply{% endif %}"
            }
        ),
        encoding="utf-8"
    )
    weights = AutoModelForCausalLM.from_config(
        Qwen2Config(
            vocab_size=4, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=context, bos_token_id=2, eos_token_id=1
        )
    )
    weights.generation_config.repetition_penalty = repetition
    return QwenModel(
        torch.device("cpu"),
        tokens,
        AutoTokenizer.from_pretrained(
            directory, local_files_only=True
        ),
        weights
    )
