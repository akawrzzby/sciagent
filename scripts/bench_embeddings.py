"""Benchmark the dense encoder on CPU and GPU.

Indexing the whole competition corpus is the operation that has to be fast
enough to stay out of the demo's way, so this script measures the real
throughput for a batch of representative passages.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

MODEL_ROOT = Path(os.environ.get("SCIAGENT_MODEL_DIR", Path.home() / "models"))
DENSE = MODEL_ROOT / "bge-m3"

PASSAGES = [
    "We propose a new simple network architecture, the Transformer, based solely on "
    "attention mechanisms, dispensing with recurrence and convolutions entirely.",
    "The Transformer allows for significantly more parallelization during training.",
    "In this work we present the Transformer, a sequence transduction model based on "
    "complex recurrent or convolutional neural networks including an encoder and decoder.",
    "We obtained 28.4 BLEU and 41.8 BLEU on the WMT 2014 English-to-German translation task.",
    "The base model was trained on 8 P100 GPUs for 12.5 days, the big model for 3.5 days.",
    "Table 2 shows the BLEU scores of the base and big models on the test sets.",
    "We use Adam with beta1 = 0.9, beta2 = 0.98 and epsilon 1e-9.",
    "The learning rate schedule is described in equation 2, with warmup steps proportional "
    "to the size of the source sequence.",
    "We compare against the state of the art in the literature and achieve a 2.3 BLEU improvement.",
    "The paper concludes with a discussion of limitations and future work directions.",
] * 6


def bench(device: str, threads: int | None, batch_size: int, seq_len: int) -> float | None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"  [skip] {device}: CUDA not available")
        return None
    if threads:
        torch.set_num_threads(threads)

    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(str(DENSE), device=device)
    model.max_seq_length = seq_len
    load_s = time.time() - t0

    # Warm-up so that the first-call CUDA/kernel-selection cost is excluded.
    model.encode(PASSAGES[:4], batch_size=batch_size, show_progress_bar=False)

    t0 = time.time()
    model.encode(PASSAGES, batch_size=batch_size, show_progress_bar=False)
    dt = time.time() - t0
    rate = len(PASSAGES) / dt
    print(f"  {device:<8} threads={threads or 'auto':<5} seq={seq_len:<4} "
          f"batch={batch_size:<4} -> {rate:7.1f} passages/s "
          f"({load_s:.1f}s load, {dt:.1f}s for {len(PASSAGES)})")
    return rate


def main() -> int:
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"cpu_threads={torch.get_num_threads()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"  gpu{i}: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB")

    print("\n=== CPU ===")
    bench("cpu", None, 32, 512)
    bench("cpu", 32, 32, 512)
    bench("cpu", 64, 32, 512)
    bench("cpu", None, 32, 256)

    if torch.cuda.is_available():
        print("\n=== GPU ===")
        bench("cuda", None, 32, 512)
        bench("cuda", None, 64, 512)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
