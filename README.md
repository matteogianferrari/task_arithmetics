# Task Arithmetic with Hugging Face CLIP

A PyTorch project for fine-tuning CLIP ViT image encoders on the eight main
classification tasks in [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089),
combining their task vectors, and evaluating the resulting models.

The trainable model contains the **vision transformer and visual projection**.
Each task has a **frozen, prompt-initialized CLIP classifier**. The pretrained text
encoder runs only during classifier preparation; training, editing, and evaluation
use image classification. No NLP training tasks are included.

This is an independent Hugging Face implementation. It follows the paper's
fine-tuning hyperparameters and arithmetic definitions, but does not claim to
reproduce its reported numbers. Dataset partition choices, compact prompt
ensembles, preprocessing, and optional mixed precision can affect results.

## 1. Linux and CUDA installation

Install [Miniconda](https://docs.anaconda.com/miniconda/) or another Conda distribution,
then run these commands from the extracted project directory:

```bash
conda env create -f environment.yml
conda activate task-arithmetic
python -m pip install --no-deps -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
ta-clip --help
```

The environment uses Python 3.11, PyTorch 2.5.1, torchvision 0.20.1, Transformers
4.49.0, and CUDA 12.4. Conda installs PyTorch and its CUDA runtime; pip installs
Transformers and the local project. This pairing is listed in the
[official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v251).
A compatible NVIDIA driver must be installed on the host. A separately installed
CUDA toolkit is unnecessary for these prebuilt packages.

For a CPU environment:

```bash
conda env create -f environment-cpu.yml
conda activate task-arithmetic-clip-cpu
python -m pip install --no-deps -e .
```

Set `device: cpu`, `precision: fp32`, and preferably `num_workers: 0` in your
experiment YAML when using CPU. CPU training is intended for checks and small
experiments.

## 2. Choose the backbone and experiment directory

| Configuration | Hugging Face model | Microbatch × accumulation | Effective batch |
| --- | --- | --- | --- |
| `configs/vit_b32.yaml` | `openai/clip-vit-base-patch32` | 32 × 4 | 128 |
| `configs/vit_b16.yaml` | `openai/clip-vit-base-patch16` | 16 × 8 | 128 |
| `configs/vit_l14.yaml` | `openai/clip-vit-large-patch14` | 8 × 16 | 128 |

The paper's Appendix B.1 specifies 2,000 optimizer updates, learning rate
`1e-5`, AdamW weight decay `0.1`, and a cosine schedule with 200 warm-up
updates. Those are the defaults. Heads and logit scale are frozen.

The YAML file also controls `data_root`, `output_root`, seeds, loader workers,
gradient clipping, checkpoint intervals, and computation precision.
Paths are relative to the directory where you run the command.

For limited GPU memory, reduce `batch_size` and increase
`accumulation_steps` while keeping their product at 128. Gradient
checkpointing is enabled. You may set `precision: bf16` on supported GPUs or
`precision: fp16` with gradient scaling. Parameters, saved backbones, and task
vectors remain float32. Defaults use fp32. The trainer runs on one GPU;
distributed training is not implemented.

Use a different `output_root` for each model, split scheme, seed, or prompt
experiment. Every task starts from `base.pt`, independently of other tasks.
Initialize that base once:

```bash
ta-clip --config configs/vit_b32.yaml init
```

You may use a local Hugging Face CLIP directory as `model_id`, or pin
`revision` to a model commit. The saved base records the resolved revision,
vision architecture, shared projection dimension, image preprocessing, and
a hash of the exact pretrained weights. Saved backbones reconstruct locally
without accessing Hugging Face.

## 3. Classification datasets

| Task | Classes | Built-in source | Split policy |
| --- | ---: | --- | --- |
| Cars | 196 | torchvision StanfordCars, local archives | Official train/test; reserve 10% of train for validation |
| DTD | 47 | torchvision DTD | Official train/val/test, partition 1 |
| EuroSAT | 10 | torchvision EuroSAT RGB | Seeded stratified 70% train / 10% val / 20% test |
| GTSRB | 43 | torchvision GTSRB | Official train/test; reserve 10% of train for validation |
| MNIST | 10 | torchvision MNIST | Official train/test; reserve 10% of train for validation |
| RESISC45 | 45 | Extracted NWPU-RESISC45 or ImageFolder | Seeded stratified 70% train / 10% val / 20% test for an unsplit archive |
| SUN397 | 397 | torchvision SUN397 plus official partition files | Official partition 1 train/test; reserve 10% of train for validation |
| SVHN | 10 | torchvision SVHN | Official train/test; reserve 10% of train for validation; extra split unused |

The table describes this project's defaults. EuroSAT and RESISC45 partition
indices are project-generated, and may differ from the authors' experiment
assets. `split_seed`, `validation_fraction`, `test_fraction`, `sun_split`,
and `dtd_split` are configurable.

With `download: true`, torchvision downloads supported datasets when missing.
**Cars and RESISC45 require manual preparation. SUN397 also requires its
official Partitions text files.** See [docs/DATASETS.md](docs/DATASETS.md) for
the exact directory layouts, sources, and explicit-manifest format.

Manifests and `train/val/test` ImageFolder layouts override built-in sources.
These routes also support custom image classification tasks.
Label order is validated before loading a saved head, and image identifiers
must be disjoint across splits. The exact assignments are saved in
`output_root/splits/<Task>.json`. Changing data or partitions in an existing
experiment is rejected. Identities cover paths, labels, and assignments;
image file contents themselves are not hashed.

First try two automatically downloadable tasks:

```bash
ta-clip --config configs/vit_b32.yaml prepare --tasks MNIST SVHN
```

This creates their dataset partitions and classifier checkpoints in
`output_root/heads`. Prototypes average normalized text embeddings and then
normalize the resulting class vector. Class names follow the dataset's label
order, including explicit digit and traffic-sign names.

The default prompt ensembles are compact and task-specific, defined in
`src/task_arithmetic_clip/heads.py`. To choose your own templates before
training:

```bash
ta-clip --config configs/vit_b32.yaml prepare --tasks MNIST SVHN \
    --templates configs/prompts.example.json
```

Each template needs exactly one `{}` placeholder. An existing experiment
keeps its prepared heads immutable. Use a fresh `output_root` to change
templates. The default preprocessing uses the saved Hugging Face CLIP
image processor and converts grayscale images to RGB. Random resized
crops and flips are optional with `augment: true`; leave them disabled for
digits and traffic signs unless that augmentation is part of your experiment.

## 4. Fine-tune independently

```bash
ta-clip --config configs/vit_b32.yaml train --tasks MNIST SVHN
```

Each task produces:

| File | Contents |
| --- | --- |
| `<Task>/finetuned.pt` | Final vision encoder and projection after the configured updates |
| `<Task>/vector.pt` | Final parameters minus the common base parameters |
| `<Task>/last.pt` | Resumable backbone, optimizer, scaler, RNG state, and consumed-batch cursor |
| `<Task>/history.jsonl` | Training loss, learning rate, and periodic validation metrics |
| `<Task>/validation.json` | Final validation accuracy, loss, and sample count |

The trainer evaluates validation data, and leaves test data for the later
evaluation command. It saves the final configured update rather than choosing
a checkpoint by test accuracy. An optimizer step skipped for fp16 overflow
does not count toward the requested training updates.

Resume one interrupted task with the same training settings:

```bash
ta-clip --config configs/vit_b32.yaml train --tasks MNIST \
    --resume runs/vit-b32/MNIST/last.pt
```

Resume preserves optimizer state and sample order even with loader prefetching.
The offline CPU test checks exact equality with uninterrupted training.
Bitwise identity across GPU hardware, kernels, worker counts, or stochastic
augmentations is not guaranteed. Existing runs require explicit resume or a
new experiment directory.

## 5. Task arithmetic

For a task `t`, the vector is:

```text
tau_t = theta_finetuned_t - theta_base
theta_edited = theta_base + lambda * sum_t(alpha_t * tau_t)
```

Arithmetic includes the vision transformer and visual projection.
Classifier heads remain separate because different tasks have different label
spaces. All vectors must have the same exact base identity and state schema.
Non-floating buffers stay equal to their base values.

**Addition** of MNIST and SVHN vectors:

```bash
ta-clip --config configs/vit_b32.yaml compose \
    --vectors runs/vit-b32/MNIST/vector.pt runs/vit-b32/SVHN/vector.pt \
    --coefficients 1 1 --scale 0.5 \
    --output runs/vit-b32/edited/mnist_svhn.pt \
    --output-vector runs/vit-b32/edited/mnist_svhn_vector.pt
```

**Negation** of the MNIST vector:

```bash
ta-clip --config configs/vit_b32.yaml compose \
    --vectors runs/vit-b32/MNIST/vector.pt \
    --coefficients -1 --scale 0.5 \
    --output runs/vit-b32/edited/negative_mnist.pt
```

**Weighted combinations and analogies** use arbitrary coefficients. For
`C + B - A`, pass three vector paths in the order `C B A` and coefficients
`1 1 -1`. The same expression is available in
[examples/arithmetic.py](examples/arithmetic.py). A useful task analogy depends
on the relationship between your datasets; the algebra alone does not ensure
improvement on a fourth task.

The Python API supports the same operations:

```python
"""Create an edited backbone from two compatible classification task vectors."""

from task_arithmetic_clip import TaskVector
from task_arithmetic_clip.checkpoints import save_payload

mnist = TaskVector.load("runs/vit-b32/MNIST/vector.pt")
svhn = TaskVector.load("runs/vit-b32/SVHN/vector.pt")
combined = mnist + svhn
save_payload(
    "runs/vit-b32/edited/combined.pt",
    combined.apply_to("runs/vit-b32/base.pt", scale=0.5),
)
```

You may also use `sum(vectors)`, `-vector`, `2.0 * vector`, and
`vector_c + vector_b - vector_a`. To extract a vector from another
fine-tuned checkpoint produced by this project:

```bash
ta-clip --config configs/vit_b32.yaml vector \
    --finetuned runs/vit-b32/MNIST/finetuned.pt \
    --output runs/vit-b32/MNIST/vector_again.pt
ta-clip inspect-vector runs/vit-b32/MNIST/vector.pt
```

Original OpenCLIP pickle checkpoints from the authors' repository are not
accepted directly. This project saves versioned tensor dictionaries and loads
them with `torch.load(..., weights_only=True)`.

## 6. Select coefficients on validation data

Search the default grid `0, 0.05, ..., 1` without retraining:

```bash
ta-clip --config configs/vit_b32.yaml sweep \
    --vectors runs/vit-b32/MNIST/vector.pt runs/vit-b32/SVHN/vector.pt \
    --tasks MNIST SVHN \
    --references MNIST=runs/vit-b32/MNIST/finetuned.pt \
                 SVHN=runs/vit-b32/SVHN/finetuned.pt \
    --output runs/vit-b32/edited/selected.pt \
    --report runs/vit-b32/metrics/addition_validation.json
```

With reference checkpoints, selection maximizes the mean of per-task accuracy
divided by the corresponding specialized fine-tuned accuracy. Without
references, it maximizes mean raw accuracy. A tie selects the smaller
coefficient. Override the grid with `--lambdas 0 0.25 0.5 0.75 1`.

Choose `--tasks` explicitly: averaging over two merged tasks measures a
different result from averaging over all eight tasks. Supply all eight
specialized references when measuring normalized accuracy over all eight.

For the paper's constrained forgetting rule, prepare ImageNet as a separate
control task using [the documented layout](docs/DATASETS.md#imagenet-control).
Then select the **largest** coefficient preserving at least 95% of the
pretrained model's validation accuracy on the control:

```bash
ta-clip --config configs/vit_b32.yaml prepare --tasks ImageNet
ta-clip --config configs/vit_b32.yaml sweep \
    --vectors runs/vit-b32/MNIST/vector.pt --coefficients -1 \
    --tasks MNIST --selection forget --control ImageNet --retain 0.95 \
    --output runs/vit-b32/edited/forget_mnist.pt \
    --report runs/vit-b32/metrics/forgetting_validation.json
```

Retention is a relative accuracy ratio. It is not a five percentage point
allowance. Validation controls selection; final target and control performance
are measured on their separate test partitions.

## 7. Compare test performance

```bash
ta-clip --config configs/vit_b32.yaml evaluate \
    --checkpoints runs/vit-b32/base.pt \
                  runs/vit-b32/MNIST/finetuned.pt \
                  runs/vit-b32/SVHN/finetuned.pt \
                  runs/vit-b32/edited/selected.pt \
    --tasks MNIST SVHN --split test \
    --references MNIST=runs/vit-b32/MNIST/finetuned.pt \
                 SVHN=runs/vit-b32/SVHN/finetuned.pt \
    --output runs/vit-b32/metrics/comparison.json
```

The command writes JSON and CSV with top-1 accuracy, cross-entropy loss,
sample count, and optional normalized accuracy. Accuracies are fractions in
`[0, 1]`; normalized accuracy can exceed one. Mean accuracy gives equal
weight to each task, regardless of dataset size.

For forgetting, evaluate `base.pt` and the selected model on
`--tasks MNIST ImageNet`. References are optional; when supplied, every
evaluated task must have a positive-accuracy specialized reference.

For all eight tasks, prepare the manual datasets first and run:

```bash
bash scripts/run_all.sh configs/vit_b32.yaml
```

The script initializes, prepares, trains, selects the global coefficient
using normalized validation accuracy across all eight tasks, and writes a
test comparison including the base, specialized models, and selected model.
It is intended for a fresh experiment directory.

## 8. Offline verification and code style

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q
ruff check .
ruff format --check .
```

Tests construct a tiny local Hugging Face CLIP checkpoint and image datasets.
They cover parent projection dimensions, frozen classifiers, fine-tuning of
both vision and projection, task-vector algebra, exact-base rejection,
held-out split separation, normalized evaluation, coefficient selection, and
checkpoint resume equality. They need no downloads or GPU.

All Python modules, classes, and functions follow the supplied style:
module descriptions, typed signatures, class Attributes sections, and
function Args and Returns sections. Comments explain implementation decisions.

See [VALIDATION.md](VALIDATION.md) for the checks executed on the delivered
project and their limits. Model weights and datasets are downloaded or
prepared by you; the archive contains source code and configuration.

## References

- [Ilharco et al., Editing Models with Task Arithmetic, ICLR 2023](https://arxiv.org/abs/2212.04089)
- [Authors' task_vectors repository](https://github.com/mlfoundations/task_vectors)
- [Hugging Face Transformers CLIP documentation, version 4.49.0](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/clip)
- [PyTorch version-specific installation instructions](https://pytorch.org/get-started/previous-versions/)
