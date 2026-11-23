from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from lib.utils import yaml2config
from networks import get_model


def main():
    parser = argparse.ArgumentParser(description="Train frequency-conditioned HiGAN")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one train batch and skip final test evaluation")
    parser.add_argument("--no-eval", action="store_true",
                        help="Explicitly skip the one-shot post-training test evaluation")
    parser.add_argument("--eval-max-samples", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml2config(args.config)
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    output = Path(args.output).resolve()
    model = get_model(cfg.model)(cfg, str(output), args.dataset_root)
    last = model.train(max_batches=1 if args.smoke_test else 0)
    if args.smoke_test:
        print(f"[train] smoke test complete; checkpoint: {last}; test split not opened", flush=True)
    elif args.no_eval:
        print(f"[train] complete; final evaluation explicitly disabled; checkpoint: {last}", flush=True)
    else:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from evaluate import evaluate
        print(f"[train] complete; evaluating last checkpoint exactly once: {last}", flush=True)
        evaluate(args.config, args.dataset_root, last, output / "evaluation.json",
                 args.eval_max_samples)


if __name__ == "__main__":
    main()
