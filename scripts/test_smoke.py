"""CPU-only smoke tests — run without a GPU (no torch/unsloth/trl import).

These guard the lab source against the most common breakages so `make test`
is a real gate, not a no-op:
- every notebook/script file exists and is valid Python (catches syntax errors)
- the TRL trainer calls use `processing_class=` (TRL >= 0.13), NOT the removed
  `tokenizer=` arg — the regression that broke NB1/NB3 on the resolved trl 0.19.x

Run:  pytest -q scripts/   (or `make test`).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "01_sft_mini", "02_preference_data", "03_dpo_train",
    "04_compare_and_eval", "05_merge_deploy_gguf", "06_benchmark",
]


def test_notebooks_exist_and_parse():
    for nb in NOTEBOOKS:
        p = REPO / "notebooks" / f"{nb}.py"
        assert p.exists(), f"missing notebook {p}"
        ast.parse(p.read_text(encoding="utf-8"))  # SyntaxError if broken


def test_scripts_parse():
    for p in (REPO / "scripts").glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8"))


def test_colab_notebooks_are_valid_json():
    for p in (REPO / "colab").glob("*.ipynb"):
        json.loads(p.read_text(encoding="utf-8"))  # ValueError if corrupt


def test_trainer_uses_processing_class_not_tokenizer():
    # TRL >= 0.13 removed the `tokenizer=` arg in favour of `processing_class=`.
    # With the requirements pin `trl>=0.12,<0.20` a fresh install resolves to
    # 0.19.x, where `DPOTrainer/SFTTrainer(tokenizer=...)` raises TypeError.
    targets = [
        "notebooks/01_sft_mini.py",
        "notebooks/03_dpo_train.py",
        "scripts/train_dpo.py",
        "colab/Lab22_DPO_T4.ipynb",
        "colab/Lab22_DPO_BigGPU.ipynb",
    ]
    offenders = [t for t in targets if "tokenizer=tokenizer" in (REPO / t).read_text(encoding="utf-8")]
    assert not offenders, (
        f"{offenders} still pass tokenizer=tokenizer to a TRL trainer; "
        f"use processing_class=tokenizer (tokenizer= removed in trl>=0.13)."
    )


def test_dpo_uses_the_sft_checkpoint_as_named_reference_adapter():
    """The reference must be frozen SFT, not the pretrained base with LoRA disabled."""
    targets = [
        "notebooks/03_dpo_train.py",
        "scripts/train_dpo.py",
        "kaggle/Lab22_DPO_Kaggle_RunAll.ipynb",
    ]
    for target in targets:
        path = REPO / target
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            text = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        else:
            text = path.read_text(encoding="utf-8")
        assert 'adapter_name=\"reference\"' in text, f"{target} does not load the SFT reference"
        assert 'ref_adapter_name=\"reference\"' in text, f"{target} does not select the reference"
        assert 'model_adapter_name=\"default\"' in text, f"{target} does not select the train adapter"


def test_kaggle_notebook_is_core_only_and_single_gpu():
    path = REPO / "kaggle" / "Lab22_DPO_Kaggle_RunAll.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'CUDA_VISIBLE_DEVICES"] = "0"' in text
    assert "01_sft_mini.py" in text
    assert "04_compare_and_eval.py" in text
    assert "05_merge_deploy_gguf.py" not in text
    assert "lab22_core_artifacts" in text
    assert "set -euo pipefail" in text
    assert "transformers==4.56.2" in text
    assert "trl==0.22.2" in text
    assert "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated" in text
    assert 'chat_template="qwen-2.5"' in text
    assert "5CD-AI/Vietnamese-alpaca-cleaned" not in text
    probe = "".join(notebook["cells"][5]["source"])
    assert probe.index("from unsloth import FastLanguageModel") < probe.index("import torch")
