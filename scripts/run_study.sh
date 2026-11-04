#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
python download_dataset.py
python scripts/download_noise.py
python -m saudasach.cli audit
python -m saudasach.cli features
python -m saudasach.cli train
python -m saudasach.cli export
# Explicit final-test access: this line is a final frozen evaluation, not tuning.
python -m saudasach.cli evaluate --final-test
python -m saudasach.cli benchmark
