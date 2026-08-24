#!/usr/bin/env python3
"""Build the single-file Kaggle T4 notebook from the four core Jupytext sources."""

from __future__ import annotations

from pathlib import Path

import jupytext
import nbformat


REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "kaggle" / "Lab22_DPO_Kaggle_RunAll.ipynb"
SOURCES = [
    REPO / "notebooks" / "01_sft_mini.py",
    REPO / "notebooks" / "02_preference_data.py",
    REPO / "notebooks" / "03_dpo_train.py",
    REPO / "notebooks" / "04_compare_and_eval.py",
]


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source)


def code(source: str):
    return nbformat.v4.new_code_cell(source)


def cleanup_cell(stage: int):
    return code(
        f"""# Release objects from stage {stage} before loading the next model.
import gc
for _name in [
    "model", "trainer", "train_result", "inputs", "out", "ds", "ds_formatted",
    "pref", "pref_ds", "tokenizer",
]:
    globals().pop(_name, None)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU cache cleared after stage {stage}. Allocated: {{torch.cuda.memory_allocated() / 1e9:.2f}} GB")"""
    )


def main() -> None:
    cells = [
        markdown(
            """# Lab 22 — DPO Alignment · Kaggle T4 · Core NB1→NB4

Đây là file chạy hoàn chỉnh dành cho Kaggle, ghép bốn phần bắt buộc:

1. SFT-mini trên Qwen2.5-3B 4-bit
2. Chuẩn bị UltraFeedback
3. DPO với SFT adapter làm reference đúng nghĩa
4. So sánh 8 prompt và xuất artifacts

Trước khi bấm **Run All**: vào **Settings → Accelerator → GPU T4 x2** (hoặc P100)
và bật **Internet**. Notebook cố ý chỉ dùng GPU số 0; pipeline này không hỗ trợ
multi-GPU và một T4 16 GB là đủ. Thời gian dự kiến khoảng 30–60 phút tùy phiên.

Các file cần nộp được lưu dưới `/kaggle/working/lab22/` và được đóng gói thành
`/kaggle/working/lab22_core_artifacts.zip` ở cell cuối."""
        ),
        markdown("## 0. Cài môi trường Kaggle"),
        code(
            """# Chạy cell này trước mọi import torch/unsloth.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["COMPUTE_TIER"] = "T4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("Configured Kaggle for the single-GPU T4 pipeline.")"""
        ),
        code(
            """%%bash
set -euo pipefail
# Bootstrap copied from Unsloth's current official Kaggle DPO notebook.
# It deliberately keeps Kaggle's data/plotting packages instead of replacing
# the whole environment, and does not require a kernel restart.
python -m pip install -q pip3-autoremove
python -m pip install -q torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
python -m pip install -q unsloth
python -m pip install -q --no-deps --upgrade "torchao>=0.16.0"
python -m pip install -q "transformers==4.56.2"
python -m pip install -q --no-deps "trl==0.22.2"
python -m pip install -q 'openai>=2.8,<3.0'"""
        ),
        code(
            """from pathlib import Path
import os

WORK = Path("/kaggle/working/lab22")
for folder in [
    WORK / "adapters" / "sft-mini",
    WORK / "adapters" / "dpo",
    WORK / "data" / "pref",
    WORK / "data" / "eval",
    WORK / "submission" / "screenshots",
]:
    folder.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
print(f"Working directory: {Path.cwd()}")"""
        ),
        code(
            """# Unsloth must be the first ML-stack import in a fresh Kaggle kernel.
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()

import torch
assert torch.cuda.is_available(), "Kaggle Settings → Accelerator → chọn GPU rồi restart session."
gpu = torch.cuda.get_device_properties(0)
print(f"PyTorch: {torch.__version__}")
print(f"Visible GPUs: {torch.cuda.device_count()} (cố ý dùng 1 GPU)")
print(f"GPU: {gpu.name} · {gpu.total_memory / 1024**3:.1f} GiB")
assert gpu.total_memory / 1024**3 >= 14, "Notebook 3B này cần khoảng 14 GiB VRAM."

from importlib.metadata import version
for package in ["unsloth", "transformers", "trl", "peft", "datasets", "bitsandbytes"]:
    print(f"{package}: {version(package)}")

import bitsandbytes as bnb
from transformers.utils import is_bitsandbytes_available
assert is_bitsandbytes_available(check_library_only=True), "bitsandbytes install is not visible to Transformers"
assert hasattr(bnb, "nn") and hasattr(bnb.nn, "Linear4bit"), "bitsandbytes 4-bit backend is unavailable"
print("Dependency self-check PASSED (Unsloth patch + bitsandbytes 4-bit).")"""
        ),
        code(
            """# Optional judge: load the key safely from Kaggle Secrets when available.
# The pipeline automatically falls back to manual judging if the secret is absent,
# invalid, out of credit, or temporarily unreachable.
try:
    from kaggle_secrets import UserSecretsClient
    _openai_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
    if _openai_key:
        os.environ["OPENAI_API_KEY"] = _openai_key
        print("OPENAI_API_KEY loaded from Kaggle Secrets.")
except Exception:
    print("OPENAI_API_KEY not available; NB4 will use manual judge mode.")"""
        ),
    ]

    for index, source_path in enumerate(SOURCES, start=1):
        cells.append(markdown(f"---\n# Stage {index}/4 · `{source_path.name}`"))
        source_nb = jupytext.read(source_path)
        cells.extend(source_nb.cells)
        if index < len(SOURCES):
            cells.append(cleanup_cell(index))

    cells.extend(
        [
            markdown("## Hoàn tất · kiểm tra và đóng gói kết quả"),
            code(
                """from pathlib import Path
import json

required = [
    "adapters/sft-mini/adapter_config.json",
    "data/pref/train.parquet",
    "adapters/dpo/adapter_config.json",
    "adapters/dpo/dpo_metrics.json",
    "data/eval/side_by_side.jsonl",
    "data/eval/judge_results.json",
    "submission/screenshots/02-sft-loss.png",
    "submission/screenshots/03-dpo-reward-curves.png",
    "submission/screenshots/04-side-by-side-table.png",
]
missing = [item for item in required if not (WORK / item).exists()]
if missing:
    raise FileNotFoundError("Thiếu artifacts: " + ", ".join(missing))

metrics = json.loads((WORK / "adapters/dpo/dpo_metrics.json").read_text())
print("Core pipeline PASSED")
print(json.dumps(metrics, indent=2, ensure_ascii=False))"""
            ),
            code(
                """import shutil
archive = shutil.make_archive(
    "/kaggle/working/lab22_core_artifacts",
    "zip",
    root_dir="/kaggle/working",
    base_dir="lab22",
)
print(f"Download artifact bundle từ Kaggle Output: {archive}")"""
            ),
            markdown(
                """### Sau khi chạy

- Nếu không có API key, NB4 sẽ ghi kết quả judge là `tie` với nhãn `MANUAL — fill in`.
  Hãy đọc `data/eval/side_by_side.jsonl`, tự chấm 8 dòng rồi sửa cell judge và chạy
  lại hai cell tổng kết. Đây là phần nhận xét cá nhân, không nên bịa tự động.
- Dùng **Save Version → Save & Run All** để Kaggle giữ output cell.
- Tải `lab22_core_artifacts.zip` từ tab **Output**. Kaggle lưu file trong
  `/kaggle/working`; không cần tải cache model về máy."""
            ),
        ]
    )

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "gpu", "internet": True},
        },
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
