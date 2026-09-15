"""Конфигурация SASRec для VK-LSVD.

Общие настройки (недели, epochs, lr, k, max_history, min_timespent_pos, seed)
наследуются из CommonConfig; здесь объявлены только модель-специфичные поля.
"""

from dataclasses import dataclass

from common.config import CommonConfig


@dataclass
class SASRecConfig(CommonConfig):
    meta_data_root: str = "data/raw/VK-LSVD/metadata"
    emb_file: str = "item_embeddings.npz"

    # ---- модель-специфичное временное окно ----
    history_weeks: int = 2

    # ---- данные / обучение ----
    batch_size: int = 64
    num_workers: int = 0

    # ---- модель ----
    emb_width: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    @property
    def weeks_root(self) -> str:
        """Папка train-недель (неделя 25 ищется отдельно в data_dir/validation)."""
        return f"{self.data_dir}/train"