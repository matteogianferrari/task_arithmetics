#!/usr/bin/env bash
# Train the eight classification tasks and select an edited model using validation accuracy.

set -euo pipefail

config_path="${1:-configs/vit_b32.yaml}"
output_root="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["output_root"])' "$config_path")"
tasks=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)
vectors=()
references=()
checkpoints=("$output_root/base.pt")

ta-clip --config "$config_path" init
ta-clip --config "$config_path" prepare --tasks "${tasks[@]}"
ta-clip --config "$config_path" train --tasks "${tasks[@]}"

for task in "${tasks[@]}"; do
    vectors+=("$output_root/$task/vector.pt")
    references+=("$task=$output_root/$task/finetuned.pt")
    checkpoints+=("$output_root/$task/finetuned.pt")
done

ta-clip --config "$config_path" sweep \
    --vectors "${vectors[@]}" \
    --tasks "${tasks[@]}" \
    --references "${references[@]}" \
    --output "$output_root/edited/all_tasks.pt" \
    --report "$output_root/metrics/all_tasks_validation.json"

checkpoints+=("$output_root/edited/all_tasks.pt")
ta-clip --config "$config_path" evaluate \
    --checkpoints "${checkpoints[@]}" \
    --tasks "${tasks[@]}" \
    --references "${references[@]}" \
    --split test \
    --output "$output_root/metrics/all_tasks_test.json"
