#!/usr/bin/env python3
"""Run the publication-facing SPED result pipeline.

The runner delegates scientific calculations to the versioned scripts in
``experiments/``. It provides a single reproducible command, validates inputs,
uses the current Python interpreter, and records the executed commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
ALL_STAGES = ("additive", "sped", "compare", "summarize")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0)


@dataclass(frozen=True)
class Step:
    """One subprocess in the reproducibility pipeline."""

    name: str
    command: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run empirical-additive, SPED, paired-comparison and summary stages "
            "with the fixed publication protocols."
        )
    )
    parser.add_argument(
        "--adata",
        type=Path,
        default=ROOT / "data" / "Norman" / "norman_2019_full_adata.h5ad",
        help="Processed Norman AnnData described in DATA.md.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all",) + ALL_STAGES,
        default=["all"],
        help="Pipeline stages to run; selected stages always execute in canonical order.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="Values of lambda_single for the matched SPED experiment.",
    )
    parser.add_argument(
        "--main-lambda",
        type=float,
        default=1.0,
        help="SPED lambda_single used in the paired main comparison.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--effect-hidden", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--sampling", choices=("cell", "condition"), default="cell")
    parser.add_argument(
        "--device",
        help="Torch device. Defaults to cuda:0 when CUDA is available, otherwise cpu.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run seed 0, lambda_single 0 and 1, at most two epochs and 1,000 bootstraps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain SPED runs even when their per-run aggregate files exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local inputs and print commands without running them.",
    )
    return parser.parse_args(argv)


def canonical_stages(requested: Sequence[str]) -> tuple[str, ...]:
    selected = set(ALL_STAGES if "all" in requested else requested)
    return tuple(stage for stage in ALL_STAGES if stage in selected)


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def effective_settings(args: argparse.Namespace) -> dict:
    seeds = list(dict.fromkeys(args.seeds))
    lambdas = list(dict.fromkeys(args.lambdas))
    epochs = args.epochs
    n_bootstrap = args.n_bootstrap
    if args.smoke:
        seeds = [0]
        lambdas = [0.0, args.main_lambda]
        epochs = min(epochs, 2)
        n_bootstrap = min(n_bootstrap, 1000)
    return {
        "stages": canonical_stages(args.stages),
        "seeds": seeds,
        "lambdas": lambdas,
        "epochs": epochs,
        "n_bootstrap": n_bootstrap,
        "device": "cpu" if args.smoke and args.device is None else resolve_device(args.device),
        "additive_mode": "smoke" if args.smoke else "standard",
    }


def result_root(args: argparse.Namespace) -> Path:
    """Keep smoke artifacts separate from publication-scale outputs."""
    base = ROOT / "outputs"
    return base / "smoke" if args.smoke else base


def validate_args(args: argparse.Namespace, settings: dict) -> None:
    stages = settings["stages"]
    if not stages:
        raise ValueError("At least one stage is required")
    if any(seed < 0 for seed in settings["seeds"]):
        raise ValueError("Seeds must be non-negative integers")
    if settings["epochs"] < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.top_k < 1 or settings["n_bootstrap"] < 1:
        raise ValueError("top-k and n-bootstrap must be positive")
    if {"additive", "sped"}.intersection(stages) and not args.adata.is_file():
        raise FileNotFoundError(
            f"Processed AnnData not found: {args.adata}\n"
            "Download/preprocess it as described in DATA.md or pass --adata."
        )
    if "sped" in stages:
        for seed in settings["seeds"]:
            split = ROOT / "protocols" / f"norman_full_seed{seed}_by_perturbation.json"
            if not split.is_file():
                raise FileNotFoundError(f"Fixed split not found: {split}")
    if "compare" in stages and "sped" in stages:
        if not any(abs(value - args.main_lambda) < 1e-12 for value in settings["lambdas"]):
            raise ValueError(
                f"--main-lambda {args.main_lambda:g} must be included in --lambdas "
                "when SPED and comparison stages are run together"
            )


def python_command(script: str, *arguments: object) -> tuple[str, ...]:
    return (sys.executable, str(EXPERIMENTS / script), *(str(value) for value in arguments))


def build_steps(args: argparse.Namespace, settings: dict) -> list[Step]:
    stages = settings["stages"]
    seeds = settings["seeds"]
    results = result_root(args)
    steps: list[Step] = []

    if "additive" in stages:
        steps.append(
            Step(
                "empirical additive baseline",
                python_command(
                    "empirical_additive.py",
                    "--project-root", ROOT,
                    "--adata", args.adata.resolve(),
                    "--mode", settings["additive_mode"],
                    "--top-k", args.top_k,
                    "--output-dir", results / "empirical_additive",
                ),
            )
        )

    if "sped" in stages:
        command = list(
            python_command(
                "sped_loss_ablation.py",
                "--project-root", ROOT,
                "--adata", args.adata.resolve(),
                "--seeds", *seeds,
                "--lambdas", *settings["lambdas"],
                "--epochs", settings["epochs"],
                "--batch-size", args.batch_size,
                "--learning-rate", args.learning_rate,
                "--weight-decay", args.weight_decay,
                "--embedding-dim", args.embedding_dim,
                "--effect-hidden", args.effect_hidden,
                "--top-k", args.top_k,
                "--device", settings["device"],
                "--sampling", args.sampling,
                "--output-dir", results / "loss_ablation",
            )
        )
        if args.smoke:
            command.append("--smoke")
        if args.force:
            command.append("--no-resume")
        steps.append(Step("SPED matched loss experiment", tuple(command)))

    if "compare" in stages:
        steps.append(
            Step(
                "paired SPED versus empirical-additive comparison",
                python_command(
                    "compare_predictions.py",
                    "--root", ROOT,
                    "--additive-dir", results / "empirical_additive",
                    "--sped-dir", results / "loss_ablation",
                    "--output-dir", results / "statistics",
                    "--seeds", *seeds,
                    "--lambda-single", args.main_lambda,
                    "--top-k", args.top_k,
                    "--n-bootstrap", settings["n_bootstrap"],
                    "--bootstrap-seed", args.bootstrap_seed,
                ),
            )
        )

    if "summarize" in stages:
        additive_aggregate = (
            results / "empirical_additive"
            / f"aggregate_{settings['additive_mode']}.csv"
        )
        sped_aggregate = results / "loss_ablation" / "aggregate_loss_ablation.csv"
        steps.append(
            Step(
                "main metric summary",
                python_command(
                    "summarize_runs.py",
                    additive_aggregate,
                    sped_aggregate,
                    "--output", results / "tables" / "main_metrics_summary.csv",
                    "--group-by", "method", "lambda_single", "sampling",
                    "--uncertainty-source", "split",
                ),
            )
        )
    return steps


def expected_inputs(step: Step, settings: dict, args: argparse.Namespace) -> tuple[Path, ...]:
    seeds = settings["seeds"]
    results = result_root(args)
    if step.name.startswith("paired"):
        paths: list[Path] = []
        for seed in seeds:
            paths.extend(
                [
                    results / "empirical_additive"
                    / f"norman_full_seed{seed}_by_perturbation_predictions.npz",
                    results / "loss_ablation"
                    / f"split{seed}_init{seed}_lambda{args.main_lambda:g}_predictions.npz",
                ]
            )
        return tuple(paths)
    if step.name == "main metric summary":
        return (
            results / "empirical_additive"
            / f"aggregate_{settings['additive_mode']}.csv",
            results / "loss_ablation" / "aggregate_loss_ablation.csv",
        )
    return ()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_pipeline(args: argparse.Namespace) -> int:
    settings = effective_settings(args)
    validate_args(args, settings)
    steps = build_steps(args, settings)

    print(f"Project root: {ROOT}")
    print(f"AnnData:      {args.adata.resolve()}")
    print(f"Stages:       {', '.join(settings['stages'])}")
    print(f"Seeds:        {settings['seeds']}")
    print(f"Device:       {settings['device']}")
    print(f"Mode:         {'smoke' if args.smoke else 'full'}")

    if args.dry_run:
        print("\nCommands (dry run):")
        for index, step in enumerate(steps, start=1):
            print(f"{index}. {step.name}")
            print(f"   {shlex.join(step.command)}")
        return 0

    started = datetime.now(timezone.utc)
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )

    completed: list[dict] = []
    for index, step in enumerate(steps, start=1):
        missing = [path for path in expected_inputs(step, settings, args) if not path.is_file()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                f"Cannot start '{step.name}'; required outputs are missing:\n{formatted}"
            )
        print(f"\n[{index}/{len(steps)}] {step.name}", flush=True)
        print(shlex.join(step.command), flush=True)
        step_started = datetime.now(timezone.utc)
        subprocess.run(step.command, cwd=ROOT, env=environment, check=True)
        completed.append(
            {
                "name": step.name,
                "command": list(step.command),
                "started_at": step_started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    manifest = {
        "pipeline": "SPED publication results",
        "git_commit": git_commit(),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "effective_settings": settings,
        "steps": completed,
    }
    manifest_path = result_root(args) / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\nAll selected stages completed. Manifest: {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_pipeline(parse_args(argv))
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"error: stage failed with exit code {error.returncode}", file=sys.stderr)
        return error.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
