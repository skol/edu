"""Общие CLI-флаги и сборка конфига для всех моделей VK-LSVD.

Все флаги опциональны: если флаг не передан, используется значение из
dataclass-конфига. Это позволяет запускать все модели с одинаковым набором
настроек и одинаковыми именами флагов.
"""

import argparse


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Добавляет общие флаги (соответствуют полям CommonConfig)."""
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Путь к подвыборке датасета VK-LSVD")
    parser.add_argument("--start-week", type=int, default=None,
                        help="Начальная неделя тренировочного периода")
    parser.add_argument("--end-week", type=int, default=None,
                        help="Конечная неделя тренировочного периода (включительно, <= 24)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Количество эпох обучения")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--k", type=int, default=None,
                        help="Глубина топ-k")
    parser.add_argument("--min-history", type=int, default=None,
                        help="Минимальный размер позитивной истории")
    parser.add_argument("--max-history", type=int, default=None,
                        help="Максимальный размер истории (обрезание текущего сэмпла)")
    parser.add_argument("--min-timespent-pos", type=int, default=None,
                        help="Порог timespent (секунд) для позитивного примера (>=)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed воспроизводимости")
    return parser


def config_from_args(args: argparse.Namespace, config_cls):
    """Строит конфиг из argparse-результата, пропуская не переданные флаги."""
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    return config_cls(**kwargs)