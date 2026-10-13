"""Combine three classification task vectors using an analogy expression."""

import argparse
from pathlib import Path

from task_arithmetic_clip import TaskVector
from task_arithmetic_clip.checkpoints import save_payload


def main() -> None:
    """Create the edited model corresponding to C + B - A.

    Args:
        None.
    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Apply the task-vector expression C + B - A.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = TaskVector.load(args.a)
    second = TaskVector.load(args.b)
    third = TaskVector.load(args.c)
    analogy = third + second - first
    save_payload(args.output, analogy.apply_to(args.base, scale=args.scale))


if __name__ == "__main__":
    main()
