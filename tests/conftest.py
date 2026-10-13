"""Small local CLIP models and image datasets for network-free integration tests."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import (
    CLIPConfig,
    CLIPImageProcessor,
    CLIPModel,
    CLIPTextConfig,
    CLIPTokenizer,
    CLIPVisionConfig,
)
from transformers.models.clip.tokenization_clip import bytes_to_unicode

from task_arithmetic_clip.config import ExperimentConfig


@pytest.fixture
def local_clip(tmp_path: Path) -> Path:
    """Create a tiny Hugging Face CLIP checkpoint with a nondefault shared projection.

    Args:
        tmp_path: Temporary pytest directory.
    Returns:
        A local pretrained model directory with weights, tokenizer, and preprocessing.
    """
    torch.set_num_threads(1)
    torch.manual_seed(17)
    path = tmp_path / "local-clip"
    path.mkdir()
    alphabet = list(bytes_to_unicode().values())
    tokens = alphabet + [character + "</w>" for character in alphabet]
    tokens += ["<|startoftext|>", "<|endoftext|>"]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    vocab_path = path / "vocab.json"
    vocab_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    merges_path = path / "merges.txt"
    merges_path.write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(str(vocab_path), str(merges_path), model_max_length=64)
    tokenizer.save_pretrained(path)
    text = CLIPTextConfig(
        vocab_size=len(tokens),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
        bos_token_id=vocabulary["<|startoftext|>"],
        eos_token_id=vocabulary["<|endoftext|>"],
        pad_token_id=0,
    )
    vision = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=16,
        patch_size=8,
    )
    config = CLIPConfig.from_text_vision_configs(text, vision, projection_dim=12)
    model = CLIPModel(config)
    model.save_pretrained(path)
    CLIPImageProcessor(
        size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
    ).save_pretrained(path)
    return path


@pytest.fixture
def experiment(tmp_path: Path, local_clip: Path) -> ExperimentConfig:
    """Create balanced image classification data and a small CPU training configuration.

    Args:
        tmp_path: Temporary pytest directory.
        local_clip: Local tiny pretrained CLIP directory.
    Returns:
        An experiment configuration for two custom classification tasks.
    """
    data_root = tmp_path / "data"
    for task in ("ToyA", "ToyB"):
        task_root = data_root / task
        task_root.mkdir(parents=True)
        manifest = {"classes": ["red block", "blue block"]}
        for split, per_class in (("train", 6), ("val", 2), ("test", 2)):
            rows = []
            for label, color in enumerate(((220, 40, 20), (20, 40, 220))):
                for index in range(per_class):
                    name = f"{split}_{label}_{index}.png"
                    Image.new("RGB", (24, 24), color).save(task_root / name)
                    rows.append({"path": name, "label": label})
            manifest[split] = rows
        (task_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ExperimentConfig(
        model_id=str(local_clip),
        data_root=str(data_root),
        output_root=str(tmp_path / "runs"),
        device="cpu",
        num_workers=0,
        batch_size=2,
        accumulation_steps=2,
        steps=4,
        warmup_steps=1,
        learning_rate=1e-3,
        gradient_checkpointing=True,
        eval_every=2,
        save_every=2,
        download=False,
    )
