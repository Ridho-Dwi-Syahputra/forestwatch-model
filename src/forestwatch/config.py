"""Konfigurasi runtime — load YAML + resolve path.

Penggunaan:

    >>> from forestwatch.config import load_config
    >>> cfg = load_config()           # default: configs/default.yaml
    >>> cfg["training"]["batch_size"]
    8
    >>> cfg.batch_size                 # akses bracket atau dot
    8
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

# ----------------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------------

# Anggap project root = parent dari src/
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
SRC_ROOT: Path = PACKAGE_ROOT.parent
PROJECT_ROOT: Path = SRC_ROOT.parent
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH: Path = CONFIGS_DIR / "default.yaml"
DEFAULT_CLASSES_PATH: Path = CONFIGS_DIR / "classes.yaml"


class Config(dict):
    """Dict yang juga support dot-access untuk top-level key."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        if isinstance(value, Mapping) and not isinstance(value, Config):
            return Config(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load arbitrary YAML file (return plain dict)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML config tidak ditemukan: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root harus dict, dapat {type(data).__name__}: {p}")
    return data


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load default config (configs/default.yaml) atau path custom.

    Bisa di-override per-field via environment variable berformat
    ``FW_{SECTION}__{KEY}`` (uppercase). Contoh: ``FW_TRAINING__BATCH_SIZE=16``.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = load_yaml(path)
    _apply_env_overrides(data)
    return Config(data)


def load_classes(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load configs/classes.yaml."""
    path = Path(path) if path else DEFAULT_CLASSES_PATH
    return load_yaml(path)


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    """Apply FW_SECTION__KEY=value env vars (limited: top-level dict only)."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("FW_") or "__" not in env_key:
            continue
        try:
            section, key = env_key[3:].lower().split("__", 1)
        except ValueError:
            continue
        if section not in cfg or not isinstance(cfg[section], dict):
            continue
        # Best-effort type cast
        cur = cfg[section].get(key)
        try:
            if isinstance(cur, bool):
                cfg[section][key] = env_val.lower() in ("1", "true", "yes")
            elif isinstance(cur, int):
                cfg[section][key] = int(env_val)
            elif isinstance(cur, float):
                cfg[section][key] = float(env_val)
            else:
                cfg[section][key] = env_val
        except (ValueError, TypeError):
            cfg[section][key] = env_val


def get_output_dir(cfg: Config | None = None) -> Path:
    """Resolve output directory dari env atau fallback ke ``./outputs``."""
    env = os.environ.get("OUTPUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT / "outputs").resolve()


def get_data_dir(cfg: Config | None = None) -> Path:
    """Resolve data directory dari env atau fallback ke ``./data``."""
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT / "data").resolve()


def get_seed(cfg: Config | None = None) -> int:
    """Seed deterministik. ENV ``SEED`` > config ``project.seed`` > 42."""
    env = os.environ.get("SEED")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    if cfg and "project" in cfg and "seed" in cfg["project"]:
        return int(cfg["project"]["seed"])
    return 42
