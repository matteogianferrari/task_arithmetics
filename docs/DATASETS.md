# Dataset preparation

Run preparation from the project directory with the same configuration used
for training and evaluation. The default data root is `./data`.

Automatic downloads use torchvision's dataset implementations. Source
availability and dataset terms are controlled by the providers. You can
replace any adapter with a local explicit manifest or ImageFolder layout.

## Torchvision adapters

| Task | Root passed to torchvision | Local directory below that root |
| --- | --- | --- |
| MNIST | `data/MNIST` | `MNIST/raw` |
| SVHN | `data/SVHN` | `train_32x32.mat`, `test_32x32.mat` |
| GTSRB | `data/GTSRB` | `gtsrb` |
| DTD | `data/DTD` | `dtd/dtd` |
| EuroSAT | `data/EuroSAT` | `eurosat/2750` |
| Cars | `data/Cars` | `stanford_cars` |
| SUN397 | `data/SUN397` | `SUN397` |

For complete dataset format details, see the
[torchvision 0.20 dataset documentation](https://pytorch.org/vision/0.20/datasets.html).

### Stanford Cars

Torchvision's original Cars download endpoint is unavailable. Obtain the
original images and labeled MAT annotations from a dataset source you can
access, and arrange them as follows:

```text
data/Cars/stanford_cars/
    cars_train/
        00001.jpg
        ...
    cars_test/
        00001.jpg
        ...
    cars_test_annos_withlabels.mat
    devkit/
        cars_train_annos.mat
        cars_meta.mat
```

The adapter always uses `download=False` for Cars, including when the
global download setting is enabled. The test annotations must contain labels.
Class names come from `cars_meta.mat`; torchvision converts one-based MAT
class IDs to zero-based labels.

Alternatively, use an ImageFolder layout with all 196 categories and a
`classnames.json` mapping numeric or abbreviated folder IDs to car names.
This alternative must preserve the original train/test allocation if you
want comparable dataset partitions.

### RESISC45

Obtain the RGB NWPU-RESISC45 image archive from the
[dataset project](https://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html),
and extract:

```text
data/RESISC45/NWPU-RESISC45/
    airplane/
        airplane_001.jpg
        ...
    airport/
        ...
    ...
```

The unsplit adapter requires all 45 categories and creates seeded stratified
train/val/test partitions. Folder underscores are replaced with spaces for
CLIP prompts. For an external benchmark split, use a manifest or
`data/RESISC45/train`, `val`, and `test` directories.

### SUN397

Torchvision can download the large image archive. The official partition
lists must be downloaded separately from the
[SUN dataset page](https://vision.princeton.edu/projects/2010/SUN/).
Extract the Partitions archive and place the selected text files beside
the image directory:

```text
data/SUN397/
    Training_01.txt
    Testing_01.txt
    SUN397/
        ClassName.txt
        a/
            abbey/
                sun_....jpg
        ...
```

`sun_split: 1` selects files numbered 01. The training list is further split
into training and validation; the testing list is used only for final testing.
Image order is resolved through the official file lists, rather than filesystem
traversal order.

## ImageFolder overrides

To bypass a built-in adapter, provide both `train` and `test` directories:

```text
data/TaskName/
    classnames.json
    train/
        class_0/
            example.png
        class_1/
            ...
    val/
        class_0/
            ...
        class_1/
            ...
    test/
        class_0/
            ...
        class_1/
            ...
```

`val` is optional. When absent, the adapter makes a stratified validation
subset from `train`. The test split never contributes training samples.
All provided splits must use identical folder-to-label mappings. The
built-in paper task names still require their full category counts.

ImageFolder assigns labels by lexicographically sorted folder identifiers.
An optional `classnames.json` must map every identifier to a readable name:

```json
{
  "class_0": "red block",
  "class_1": "blue block"
}
```

The mapping never changes integer label order. Without the mapping, general
folder names are converted by replacing underscores and slashes with spaces.
Digit and traffic-sign folders use zero-based numeric IDs unless a complete
custom mapping is provided.

## Explicit manifest overrides

A `data/TaskName/manifest.json` file takes priority over every other layout:

```json
{
  "classes": ["red block", "blue block"],
  "train": [
    {"path": "images/train_red.png", "label": 0},
    {"path": "images/train_blue.png", "label": 1}
  ],
  "val": [
    {"path": "images/val_red.png", "label": 0},
    {"path": "images/val_blue.png", "label": 1}
  ],
  "test": [
    {"path": "images/test_red.png", "label": 0},
    {"path": "images/test_blue.png", "label": 1}
  ]
}
```

Image paths are relative to `data/TaskName`. Labels index the `classes`
list directly. Paths must be relative, and the same image cannot appear in
multiple partitions. Manifests should include your actual full sample lists;
the two-class example above illustrates the format only.

Use a custom task name for a new label space:

```bash
ta-clip --config configs/vit_b32.yaml prepare --tasks TaskName
ta-clip --config configs/vit_b32.yaml train --tasks TaskName
```

Generic photo prompts are used unless you supply custom templates. A
manifest for a paper task must include that task's full category count.
Provide enough training images for at least one complete microbatch.

## ImageNet control

The optional ImageNet control requires manually acquired ImageNet-1k images.
It does not require ImageNet fine-tuning for the negation experiment.
Organize the labeled official validation set into WNID folders:

```text
data/ImageNet/
    classnames.json
    val/
        n01440764/
            ILSVRC2012_val_....JPEG
        n01443537/
            ...
        ...
```

`classnames.json` must map all 1,000 folder identifiers to unique readable
CLIP class names, for example:

```json
{
  "n01440764": "tench",
  "n01443537": "goldfish"
}
```

The example mapping is abbreviated; your file must contain every category.
Never use WNIDs themselves as CLIP prompt class names.

For this validation-only directory layout, the project reserves a seeded
stratified fraction for coefficient-selection validation and uses the
remainder as its final control test set. With the default fraction 0.1,
each category's 50 official validation examples contributes 5 to selection
and 45 to final evaluation. There is no control training partition.

For separately defined control validation and test sets, supply a manifest
with 1,000 class names, `val`, and `test`; `train` may be omitted for
ImageNet. This makes the selected control partition explicit.

The control split and compact ImageNet prompts differ from the authors'
original evaluation assets. Document those choices when reporting results.
