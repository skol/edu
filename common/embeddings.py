"""Загрузка pretrained item-эмбеддингов из .npz."""

import os

import numpy as np


def load_embeddings_only(meta_root: str, emb_file: str, emb_dim: int):
    """Читает item_embeddings.npz и возвращает (vecs, id_to_pos)."""
    path = os.path.join(meta_root, emb_file)
    data = np.load(path)
    ids_np = data[data.files[0]]
    vecs_np = data[data.files[1]][:, :emb_dim].astype(np.float32)

    print("\nEmbedding diagnostics:")
    print("shape:", vecs_np.shape)
    print("NaN:", np.isnan(vecs_np).sum())
    print("Inf:", np.isinf(vecs_np).sum())
    print("finite:", np.isfinite(vecs_np).all())

    if not np.isfinite(vecs_np).all():
        bad_rows = np.where(~np.isfinite(vecs_np).all(axis=1))[0]
        print("Количество плохих embeddings:", len(bad_rows))
        raise RuntimeError("item_embeddings.npz contains NaN/Inf")

    id_to_pos = {int(_id): idx for idx, _id in enumerate(ids_np)}
    print(f"Загружено {len(ids_np):,} embeddings.")
    return vecs_np, id_to_pos