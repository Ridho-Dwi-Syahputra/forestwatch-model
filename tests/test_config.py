"""Test config loader."""

from __future__ import annotations

import os

from forestwatch.config import (
    DEFAULT_CLASSES_PATH,
    DEFAULT_CONFIG_PATH,
    Config,
    load_classes,
    load_config,
    load_yaml,
)


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg["project"]["name"] == "ForestWatch Papua"
    assert cfg["periods"]["t1"] == 2021
    assert cfg["periods"]["t2"] == 2025


def test_dot_access():
    cfg = load_config()
    assert cfg.project.name == "ForestWatch Papua"
    assert cfg.training.batch_size == 8


def test_classes_yaml_loads():
    classes = load_classes()
    assert "classes" in classes
    assert len(classes["classes"]) == 7
    assert classes["classes"][1]["name"] == "Hutan"


def test_env_override(monkeypatch):
    monkeypatch.setenv("FW_TRAINING__BATCH_SIZE", "16")
    cfg = load_config()
    assert cfg.training.batch_size == 16


def test_env_override_bool(monkeypatch):
    monkeypatch.setenv("FW_TRAINING__AMP", "false")
    cfg = load_config()
    assert cfg.training.amp is False


def test_default_paths_exist():
    assert DEFAULT_CONFIG_PATH.exists()
    assert DEFAULT_CLASSES_PATH.exists()
