"""Command-line workflow for classification fine-tuning and task arithmetic."""

import argparse
import json
from pathlib import Path

from task_arithmetic_clip.arithmetic import TaskVector
from task_arithmetic_clip.checkpoints import save_payload
from task_arithmetic_clip.config import ExperimentConfig
from task_arithmetic_clip.data import PAPER_TASKS
from task_arithmetic_clip.evaluation import evaluate_checkpoints
from task_arithmetic_clip.training import train_task
from task_arithmetic_clip.workflow import (
    combine_vectors,
    initialize,
    prepare_tasks,
    sweep_coefficients,
)


def parse_references(values: list[str] | None) -> dict[str, str] | None:
    """Parse specialized reference checkpoints from task=path arguments.

    Args:
        values: Optional reference arguments.

    Returns:
        A mapping from task names to checkpoint paths, or None.
    """
    if values is None:
        return None

    references = {}
    for value in values:
        if "=" not in value:
            raise ValueError("References must use Task=checkpoint.pt syntax.")

        task, path = value.split("=", 1)
        if not task or not path or task in references:
            raise ValueError("Reference tasks must be unique and paths nonempty.")

        references[task] = path

    return references


def build_parser() -> argparse.ArgumentParser:
    """Create the project's command-line argument parser.

    Args:
        None.

    Returns:
        The parser for initialization, training, editing, and evaluation commands.
    """
    parser = argparse.ArgumentParser(description="CLIP vision classification with task arithmetic.")
    parser.add_argument("--config", default="configs/vit_b32.yaml", help="Experiment YAML path.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="Download CLIP and save the common vision-only base.")

    prepare = commands.add_parser("prepare", help="Prepare datasets and frozen CLIP classifiers.")
    prepare.add_argument("--tasks", nargs="+", default=list(PAPER_TASKS))
    prepare.add_argument("--templates", help="Optional JSON mapping tasks to prompt-template lists.")

    train = commands.add_parser("train", help="Independently fine-tune each task from the base.")
    train.add_argument("--tasks", nargs="+", default=list(PAPER_TASKS))
    train.add_argument("--resume", help="Resume one task from its last.pt checkpoint.")

    vector = commands.add_parser("vector", help="Extract a task vector from a fine-tuned checkpoint.")
    vector.add_argument("--base", help="Base checkpoint, defaults to output_root/base.pt.")
    vector.add_argument("--finetuned", required=True)
    vector.add_argument("--output", required=True)

    compose = commands.add_parser("compose", help="Apply a weighted sum of compatible task vectors.")
    compose.add_argument("--base", help="Base checkpoint, defaults to output_root/base.pt.")
    compose.add_argument("--vectors", nargs="+", required=True)
    compose.add_argument("--coefficients", nargs="+", type=float)
    compose.add_argument("--scale", type=float, default=1.0)
    compose.add_argument("--output", required=True, help="Edited backbone checkpoint.")
    compose.add_argument("--output-vector", help="Optional unscaled combined vector checkpoint.")

    evaluate = commands.add_parser("evaluate", help="Evaluate local backbones with saved task heads.")
    evaluate.add_argument("--checkpoints", nargs="+", required=True)
    evaluate.add_argument("--tasks", nargs="+", default=list(PAPER_TASKS))
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--references", nargs="+", help="Specialized Task=checkpoint.pt references.")
    evaluate.add_argument("--output", help="JSON report path, also writes CSV.")

    sweep = commands.add_parser("sweep", help="Select a global coefficient using validation only.")
    sweep.add_argument("--vectors", nargs="+", required=True)
    sweep.add_argument("--coefficients", nargs="+", type=float)
    sweep.add_argument("--tasks", nargs="+", default=list(PAPER_TASKS))
    sweep.add_argument("--lambdas", nargs="+", type=float, default=[value / 20 for value in range(21)])
    sweep.add_argument("--references", nargs="+", help="Specialized Task=checkpoint.pt references.")
    sweep.add_argument("--selection", choices=["maximize", "forget"], default="maximize")
    sweep.add_argument("--control", help="Separate control task for forgetting, usually ImageNet.")
    sweep.add_argument("--retain", type=float, default=0.95)
    sweep.add_argument("--output", required=True, help="Selected edited backbone checkpoint.")
    sweep.add_argument("--report", help="Validation search JSON path.")

    inspect = commands.add_parser("inspect-vector", help="Show a vector's identity, size, and norm.")
    inspect.add_argument("path")

    return parser


def main() -> None:
    """Execute one command-line operation.

    Args:
        None.

    Returns:
        None.
    """
    parser = build_parser()

    args = parser.parse_args()
    if args.command == "inspect-vector":
        vector = TaskVector.load(args.path)
        print(
            json.dumps(
                {
                    "base_id": vector.base_id,
                    "schema_id": vector.schema_id,
                    "parameters": sum(value.numel() for value in vector.delta.values()),
                    "norm": vector.norm(),
                    "provenance": vector.provenance,
                },
                indent=2,
            )
        )
        return

    config = ExperimentConfig.load(args.config)
    base_path = Path(config.output_root) / "base.pt"
    if args.command == "init":
        print(initialize(config))
    elif args.command == "prepare":
        templates = json.loads(Path(args.templates).read_text()) if args.templates else None
        for path in prepare_tasks(args.tasks, config, templates):
            print(path)
    elif args.command == "train":
        if args.resume and len(args.tasks) != 1:
            parser.error("--resume requires exactly one --tasks entry.")
        if len(set(args.tasks)) != len(args.tasks):
            parser.error("Training task names must be unique.")
        for task in args.tasks:
            print(train_task(task, config, args.resume))
    elif args.command == "vector":
        vector = TaskVector.from_checkpoints(args.base or base_path, args.finetuned)
        vector.save(args.output)
        print(args.output)
    elif args.command == "compose":
        if Path(args.output).resolve() == Path(args.base or base_path).resolve():
            raise ValueError("The edited output must differ from the common pretrained base.")
        vector = combine_vectors(args.vectors, args.coefficients)
        save_payload(args.output, vector.apply_to(args.base or base_path, args.scale))
        if args.output_vector:
            vector.save(args.output_vector)
        print(args.output)
    elif args.command == "evaluate":
        report = evaluate_checkpoints(
            args.checkpoints,
            args.tasks,
            config,
            args.split,
            args.output or Path(config.output_root) / "metrics" / "evaluation.json",
            parse_references(args.references),
        )
        for result in report["results"]:
            print(f"{result['checkpoint']}: mean_accuracy={result['mean_accuracy']:.5f}")
    elif args.command == "sweep":
        vector = combine_vectors(args.vectors, args.coefficients)
        report = sweep_coefficients(
            vector,
            args.tasks,
            args.lambdas,
            config,
            args.output,
            args.report or Path(config.output_root) / "metrics" / "sweep.json",
            parse_references(args.references),
            args.selection,
            args.control,
            args.retain,
        )
        print(f"selected_lambda={report['selected_lambda']:.3f} checkpoint={args.output}")
