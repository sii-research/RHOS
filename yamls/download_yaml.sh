#!/bin/bash

# 定义任务列表
TASKS=("pusht" "square_mh" "square_ph" "tool_hang_ph" "transport_mh" "transport_ph")

# 基础 URL
BASE_URL="https://diffusion-policy.cs.columbia.edu/data/experiments/image"

# 遍历任务列表
for task in "${TASKS[@]}"; do
    FILENAME="${task}.yaml"
    URL="${BASE_URL}/${task}/diffusion_policy_cnn/config.yaml"
    echo "Downloading ${URL} -> ${FILENAME}"
    wget -O "${FILENAME}" "${URL}"
done