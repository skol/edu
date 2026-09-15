"""SASRec (Self-Attentive Sequential Recommendation) на PyTorch для VK-LSVD."""

from .config import SASRecConfig
from .data import (
    SequentialDataset,
    TrainIterableDataset,
    build_sliding_samples,
    collate_fn,
    extract_week_interactions,
    load_week_df,
    read_week_item_ids,
    remap_sample,
    remap_samples,
)
from .embeddings import build_weight_matrix, load_embeddings_only
from .model import SASRec
from .train import evaluate, get_device, seed_everything, train_epoch

__all__ = [
    "SASRecConfig",
    "SequentialDataset",
    "TrainIterableDataset",
    "build_sliding_samples",
    "collate_fn",
    "extract_week_interactions",
    "load_week_df",
    "read_week_item_ids",
    "remap_sample",
    "remap_samples",
    "build_weight_matrix",
    "load_embeddings_only",
    "SASRec",
    "evaluate",
    "get_device",
    "seed_everything",
    "train_epoch",
]