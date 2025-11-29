#!/bin/bash

# download original yaml files for all tasks from diffusion-policy repo

# task_list
TASKS=("pusht" "square_mh" "square_ph" "tool_hang_ph" "transport_mh" "transport_ph")

# base_url
BASE_URL="https://diffusion-policy.cs.columbia.edu/data/experiments/image"

# download each yaml file
for task in "${TASKS[@]}"; do
    FILENAME="${task}.yaml"
    URL="${BASE_URL}/${task}/diffusion_policy_cnn/config.yaml"
    echo "Downloading ${URL} -> ${FILENAME}"
    wget -O "${FILENAME}" "${URL}"
done