"""Test model card renderer."""

from __future__ import annotations

from forestwatch.outputs.model_card import render_model_card


def test_renders_minimal(tmp_path):
    out = tmp_path / "model_card.md"
    saved = render_model_card(out)
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    # Section wajib
    assert "# Model Card" in content
    assert "## Ringkasan" in content
    assert "## Data Latih" in content
    assert "## Hasil Evaluasi" in content
    assert "## Keterbatasan" in content
    assert "## Penggunaan yang Dimaksud" in content
    assert "## Sitasi" in content


def test_renders_with_metrics(tmp_path, sample_metrics):
    out = tmp_path / "model_card.md"
    render_model_card(out, metrics=sample_metrics, epochs=50, batch_size=8, n_parameters=32_000_000)
    content = out.read_text(encoding="utf-8")
    assert "32,000,000" in content  # parameters formatted
    assert "0.7100" in content or "0.7100" in content  # mean_iou
    # Per-class table
    assert "| Perairan |" in content


def test_renders_with_extra_sections(tmp_path):
    out = tmp_path / "model_card.md"
    render_model_card(out, extra_sections={"Catatan": "Ini test."})
    assert "## Catatan" in out.read_text(encoding="utf-8")
