"""Classification datasets, deterministic partitions, and CLIP preprocessing."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transformers import CLIPImageProcessor

from task_arithmetic_clip.checkpoints import write_json
from task_arithmetic_clip.config import ExperimentConfig

PAPER_TASKS = ("Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN")
CLASS_COUNTS = dict(zip(PAPER_TASKS, (196, 47, 10, 43, 10, 45, 397, 10))) | {"ImageNet": 1000}
DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
EUROSAT_NAMES = {
    "AnnualCrop": "annual crop land",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation land",
    "Highway": "highway",
    "Industrial": "industrial buildings",
    "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land",
    "Residential": "residential buildings",
    "River": "river",
    "SeaLake": "sea or lake",
}
GTSRB_NAMES = (
    "speed limit 20 km/h",
    "speed limit 30 km/h",
    "speed limit 50 km/h",
    "speed limit 60 km/h",
    "speed limit 70 km/h",
    "speed limit 80 km/h",
    "end of speed limit 80 km/h",
    "speed limit 100 km/h",
    "speed limit 120 km/h",
    "no passing",
    "no passing for vehicles over 3.5 metric tons",
    "right of way at the next intersection",
    "priority road",
    "yield",
    "stop",
    "no vehicles",
    "vehicles over 3.5 metric tons prohibited",
    "no entry",
    "general caution",
    "dangerous curve to the left",
    "dangerous curve to the right",
    "double curve",
    "bumpy road",
    "slippery road",
    "road narrows on the right",
    "road work",
    "traffic signals",
    "pedestrians",
    "children crossing",
    "bicycles crossing",
    "beware of ice or snow",
    "wild animals crossing",
    "end of all speed and passing limits",
    "turn right ahead",
    "turn left ahead",
    "ahead only",
    "go straight or right",
    "go straight or left",
    "keep right",
    "keep left",
    "roundabout mandatory",
    "end of no passing",
    "end of no passing for vehicles over 3.5 metric tons",
)


class ImagePreprocessor:
    """ImagePreprocessor class for using saved Hugging Face CLIP image settings.

    Attributes:
        processor: CLIP image processor reconstructed without network access.
        augmentation: Optional random image augmentation before CLIP preprocessing.
    """

    def __init__(self, config: dict[str, Any], augment: bool = False) -> None:
        """Initialize preprocessing from checkpoint metadata.

        Args:
            config: Serialized Hugging Face image processor configuration.
            augment: Whether to use random crops and horizontal flips.
        Returns:
            None.
        """
        self.processor = CLIPImageProcessor.from_dict(config)
        size = self.processor.crop_size["height"]
        self.augmentation = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        size, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            if augment
            else None
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Convert an image into a normalized CLIP pixel tensor.

        Args:
            image: PIL image in any mode, including grayscale.
        Returns:
            A float32 tensor of shape [3, height, width].
        """
        image = image.convert("RGB")
        if self.augmentation is not None:
            image = self.augmentation(image)
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


class DatasetView(Dataset):
    """DatasetView class for applying preprocessing to a fixed subset.

    Attributes:
        dataset: Dataset returning unprocessed images and integer labels.
        indices: Indices belonging to this partition.
        transform: Image preprocessing callable.
    """

    def __init__(self, dataset: Dataset, indices: list[int], transform: ImagePreprocessor) -> None:
        """Initialize a dataset partition.

        Args:
            dataset: Source image dataset.
            indices: Ordered indices included in the partition.
            transform: CLIP image preprocessing.
        Returns:
            None.
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of examples in the partition.

        Args:
            None.
        Returns:
            The number of selected examples.
        """
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load and preprocess an example.

        Args:
            index: Index within this partition.
        Returns:
            The normalized image tensor and its integer class label.
        """
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), int(label)


class ManifestDataset(Dataset):
    """ManifestDataset class for classification data with explicit split lists.

    Attributes:
        root: Directory relative to which image paths are interpreted.
        rows: Validated image paths and labels.
        targets: Integer labels in manifest order.
    """

    def __init__(self, root: Path, rows: list[dict[str, Any]], count: int) -> None:
        """Initialize a manifest-backed dataset.

        Args:
            root: Dataset directory.
            rows: Records containing relative path and integer label fields.
            count: Number of classification categories.
        Returns:
            None.
        """
        self.root = root.resolve()
        self.rows = []
        for row in rows:
            path = self.root / row["path"]
            label = row["label"]
            if (
                Path(row["path"]).is_absolute()
                or not path.resolve().is_relative_to(self.root)
                or not isinstance(label, int)
                or not 0 <= label < count
            ):
                raise ValueError("Manifest paths must be relative and labels valid class indices.")
            self.rows.append({"path": path.relative_to(self.root).as_posix(), "label": label})
        self.targets = [row["label"] for row in self.rows]

    def __len__(self) -> int:
        """Return the number of manifest records.

        Args:
            None.
        Returns:
            The number of images.
        """
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        """Open a manifest image in RGB mode.

        Args:
            index: Index within the manifest.
        Returns:
            The RGB image and its integer class label.
        """
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            return image.convert("RGB"), row["label"]


@dataclass
class DatasetBundle:
    """DatasetBundle class for classification splits and label metadata.

    Attributes:
        train: Training partition, or None for a control-only dataset.
        val: Held-out validation partition.
        test: Held-out test partition.
        class_names: Human-readable class names in exact label order.
        split_id: Identity of class names, sample identifiers, and split assignments.
    """

    train: DatasetView | None
    val: DatasetView
    test: DatasetView
    class_names: list[str]
    split_id: str


def stratified_partition(
    labels: list[int], fractions: tuple[float, ...], seed: int
) -> list[list[int]]:
    """Partition examples deterministically while retaining each category in every split.

    Args:
        labels: Integer class labels for the source examples.
        fractions: Positive fractions summing to one, in output partition order.
        seed: Random seed used only for partitioning.
    Returns:
        Sorted, disjoint index lists covering the complete source dataset.
    """
    if not fractions or any(value <= 0 for value in fractions):
        raise ValueError("Partition fractions must be positive.")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Partition fractions must sum to one.")
    rng = np.random.default_rng(seed)
    outputs: list[list[int]] = [[] for _ in fractions]
    array = np.asarray(labels)
    for label in sorted(set(labels)):
        indices = np.flatnonzero(array == label)
        if len(indices) < len(fractions):
            raise ValueError(f"Class {label} has too few samples for the requested partitions.")
        rng.shuffle(indices)
        counts = np.floor(np.asarray(fractions) * len(indices)).astype(int)
        counts = np.maximum(counts, 1)
        while counts.sum() > len(indices):
            candidates = np.where(counts > 1)[0]
            counts[candidates[np.argmax(counts[candidates])]] -= 1
        # Allocate remaining examples to the partition furthest below its requested count.
        desired = np.asarray(fractions) * len(indices)
        while counts.sum() < len(indices):
            counts[np.argmax(desired - counts)] += 1
        start = 0
        for output, count in zip(outputs, counts):
            output.extend(indices[start : start + count].tolist())
            start += count
    return [sorted(output) for output in outputs]


def dataset_labels(dataset: Dataset) -> list[int]:
    """Read labels from supported datasets without opening their images.

    Args:
        dataset: Torchvision or manifest classification dataset.
    Returns:
        Integer labels in dataset order.
    """
    for attribute in ("targets", "labels", "_labels"):
        if hasattr(dataset, attribute):
            return [int(value) for value in getattr(dataset, attribute)]
    for attribute in ("samples", "_samples"):
        if hasattr(dataset, attribute):
            return [int(row[1]) for row in getattr(dataset, attribute)]
    raise TypeError(f"Unsupported label metadata for {type(dataset).__name__}.")


def sample_ids(dataset: Dataset, prefix: str, root: Path) -> list[str]:
    """Create portable identifiers for dataset examples.

    Args:
        dataset: Source dataset.
        prefix: Partition source identifier for datasets without individual image files.
        root: Global dataset root used to make image paths relative.
    Returns:
        Sample identifiers in source dataset order.
    """
    if isinstance(dataset, ManifestDataset):
        return [
            (dataset.root / row["path"]).resolve().relative_to(root.resolve()).as_posix()
            for row in dataset.rows
        ]
    paths = getattr(dataset, "_image_files", None)
    if paths is None:
        samples = getattr(dataset, "samples", getattr(dataset, "_samples", None))
        paths = [row[0] for row in samples] if samples is not None else None
    if paths is not None:
        return [Path(path).resolve().relative_to(root.resolve()).as_posix() for path in paths]
    return [f"{prefix}:{index}" for index in range(len(dataset))]


def folder_class_names(task: str, root: Path, classes: list[str]) -> list[str]:
    """Resolve folder labels into human-readable CLIP class names.

    Args:
        task: Classification task name.
        root: Dataset directory containing an optional classnames.json mapping.
        classes: Folder identifiers in ImageFolder label order.
    Returns:
        Class names in the same order as the folder identifiers.
    """
    mapping_path = root / "classnames.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict) or set(mapping) != set(classes):
            raise ValueError("classnames.json must map every folder identifier exactly once.")
        return [mapping[name] for name in classes]
    if task == "ImageNet":
        raise ValueError("ImageNet requires classnames.json mapping WNIDs to readable class names.")
    if task in {"MNIST", "SVHN", "GTSRB"}:
        ids = [int(name) for name in classes]
        if sorted(ids) != list(range(len(classes))):
            raise ValueError(f"{task} folders must use zero-based numeric labels.")
        names = GTSRB_NAMES if task == "GTSRB" else DIGITS
        return [names[index] for index in ids]
    if task == "EuroSAT":
        return [EUROSAT_NAMES.get(name, name.replace("_", " ")) for name in classes]
    return [name.replace("_", " ").replace("/", " ") for name in classes]


def build_dataset(
    task: str, config: ExperimentConfig, processor_config: dict[str, Any]
) -> DatasetBundle:
    """Build classification splits and persist their exact sample assignments.

    Explicit manifests and train/val/test ImageFolder layouts take precedence over
    torchvision adapters. Existing split metadata must agree with the current data.

    Args:
        task: Paper task, ImageNet control task, or a custom manifest task.
        config: Dataset and experiment settings.
        processor_config: Image processor configuration saved in the base checkpoint.
    Returns:
        Preprocessed training, validation, and test datasets with consistent class labels.
    """
    if Path(task).name != task or task in {".", ".."}:
        raise ValueError("Task names must be single directory names.")
    root = Path(config.data_root)
    task_root = root / task
    sources: dict[str, tuple[Dataset, list[int], str]] = {}
    manifest_path = task_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        class_names = manifest["classes"]
        for split in ("train", "val", "test"):
            rows = manifest.get(split, [])
            if not rows and split == "train" and task == "ImageNet":
                continue
            if not rows:
                raise ValueError(f"The {task} manifest has no {split} examples.")
            raw = ManifestDataset(task_root, rows, len(class_names))
            sources[split] = (raw, list(range(len(raw))), split)
    elif (task_root / "train").is_dir() and (task_root / "test").is_dir():
        training = datasets.ImageFolder(task_root / "train")
        testing = datasets.ImageFolder(task_root / "test")
        class_names = folder_class_names(task, task_root, training.classes)
        if testing.class_to_idx != training.class_to_idx:
            raise ValueError(f"{task} train and test folders have different class mappings.")
        sources["test"] = (testing, list(range(len(testing))), "test")
        if (task_root / "val").is_dir():
            validation = datasets.ImageFolder(task_root / "val")
            if validation.class_to_idx != training.class_to_idx:
                raise ValueError(f"{task} validation folders have a different class mapping.")
            sources["train"] = (training, list(range(len(training))), "train")
            sources["val"] = (validation, list(range(len(validation))), "val")
        else:
            train_ids, val_ids = stratified_partition(
                dataset_labels(training),
                (1 - config.validation_fraction, config.validation_fraction),
                config.split_seed,
            )
            sources["train"] = (training, train_ids, "train")
            sources["val"] = (training, val_ids, "train")
    elif task == "ImageNet" and (task_root / "val").is_dir():
        raw = datasets.ImageFolder(task_root / "val")
        class_names = folder_class_names(task, task_root, raw.classes)
        val_ids, test_ids = stratified_partition(
            dataset_labels(raw),
            (config.validation_fraction, 1 - config.validation_fraction),
            config.split_seed,
        )
        sources["val"] = (raw, val_ids, "official-val")
        sources["test"] = (raw, test_ids, "official-val")
    elif task in {"EuroSAT", "RESISC45"}:
        if task == "EuroSAT":
            raw = datasets.EuroSAT(task_root, download=config.download)
        else:
            raw_root = task_root / "NWPU-RESISC45"
            if not raw_root.is_dir():
                raise FileNotFoundError(
                    f"Extract RESISC45 into {raw_root}, or supply train/test folders or a manifest."
                )
            raw = datasets.ImageFolder(raw_root)
        class_names = folder_class_names(task, task_root, raw.classes)
        train_ids, val_ids, test_ids = stratified_partition(
            dataset_labels(raw),
            (
                1 - config.validation_fraction - config.test_fraction,
                config.validation_fraction,
                config.test_fraction,
            ),
            config.split_seed,
        )
        sources = {
            "train": (raw, train_ids, "all"),
            "val": (raw, val_ids, "all"),
            "test": (raw, test_ids, "all"),
        }
    elif task == "DTD":
        raw_splits = {
            split: datasets.DTD(
                task_root, split=split, partition=config.dtd_split, download=config.download
            )
            for split in ("train", "val", "test")
        }
        class_names = list(raw_splits["train"].classes)
        for split, raw in raw_splits.items():
            if list(raw.classes) != class_names:
                raise ValueError("DTD split class order differs.")
            sources[split] = (raw, list(range(len(raw))), split)
    elif task == "SUN397":
        raw = datasets.SUN397(task_root, download=config.download)
        class_names = [
            " ".join(reversed(name.split("/"))).replace("_", " ") for name in raw.classes
        ]
        path_to_index = {
            path.relative_to(task_root / "SUN397").as_posix(): index
            for index, path in enumerate(raw._image_files)
        }
        official = {}
        for name, prefix in (("train", "Training"), ("test", "Testing")):
            path = task_root / f"{prefix}_{config.sun_split:02d}.txt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Extract the official SUN397 Partitions files "
                    f"into {task_root}."
                )
            lines = [line.strip().lstrip("/") for line in path.read_text().splitlines()]
            if any(line not in path_to_index for line in lines):
                raise ValueError(f"{path} references images absent from the SUN397 archive.")
            official[name] = [path_to_index[line] for line in lines]
        labels = dataset_labels(raw)
        train_local, val_local = stratified_partition(
            [labels[index] for index in official["train"]],
            (1 - config.validation_fraction, config.validation_fraction),
            config.split_seed,
        )
        sources = {
            "train": (raw, [official["train"][index] for index in train_local], "all"),
            "val": (raw, [official["train"][index] for index in val_local], "all"),
            "test": (raw, official["test"], "all"),
        }
    elif task in {"MNIST", "SVHN", "GTSRB", "Cars"}:
        if task == "MNIST":
            training = datasets.MNIST(task_root, train=True, download=config.download)
            testing = datasets.MNIST(task_root, train=False, download=config.download)
            class_names = list(DIGITS)
        elif task == "SVHN":
            training = datasets.SVHN(task_root, split="train", download=config.download)
            testing = datasets.SVHN(task_root, split="test", download=config.download)
            class_names = list(DIGITS)
        elif task == "GTSRB":
            training = datasets.GTSRB(task_root, split="train", download=config.download)
            testing = datasets.GTSRB(task_root, split="test", download=config.download)
            class_names = list(GTSRB_NAMES)
        else:
            # Torchvision's Stanford Cars download URL is unavailable; use local archives.
            training = datasets.StanfordCars(task_root, split="train", download=False)
            testing = datasets.StanfordCars(task_root, split="test", download=False)
            class_names = list(training.classes)
        train_ids, val_ids = stratified_partition(
            dataset_labels(training),
            (1 - config.validation_fraction, config.validation_fraction),
            config.split_seed,
        )
        sources = {
            "train": (training, train_ids, "train"),
            "val": (training, val_ids, "train"),
            "test": (testing, list(range(len(testing))), "test"),
        }
    else:
        raise ValueError(
            f"Unknown task {task}. Supply a manifest or train/test ImageFolder layout."
        )
    if (
        not isinstance(class_names, list)
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError("Class names must be unique, nonempty strings in label order.")
    if task in CLASS_COUNTS and len(class_names) != CLASS_COUNTS[task]:
        raise ValueError(f"{task} requires {CLASS_COUNTS[task]} classes, found {len(class_names)}.")
    assignments = {}
    views = {}
    used = set()
    for split, (raw, indices, prefix) in sources.items():
        ids = sample_ids(raw, f"{task}:{prefix}", root)
        labels = dataset_labels(raw)
        records = [[ids[index], labels[index]] for index in indices]
        current = {record[0] for record in records}
        if len(current) != len(records) or used & current:
            raise ValueError(f"{task} contains repeated examples or overlapping partitions.")
        used.update(current)
        assignments[split] = records
        views[split] = DatasetView(
            raw, indices, ImagePreprocessor(processor_config, config.augment and split == "train")
        )
    manifest = {
        "task": task,
        "class_names": class_names,
        "assignments": assignments,
        "split_seed": config.split_seed,
    }
    split_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["split_id"] = split_id
    destination = Path(config.output_root) / "splits" / f"{task}.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                f"{task} data or partitions changed. Use a new output_root for the new experiment."
            )
    else:
        write_json(destination, manifest)
    return DatasetBundle(views.get("train"), views["val"], views["test"], class_names, split_id)
