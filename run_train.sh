#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
#
# COMM_MODE options for debugging:
#
# 1. "fake_backend" - Dry-run mode for config validation without GPU execution
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=32 COMM_MODE="fake_backend" ./run_train.sh
#
# 2. "local_tensor" - Single-GPU debugging mode with simulated multi-GPU behavior
#    - All communication and computation execute on a single shared GPU
#    - Simulates the full training workflow without actual distributed communication
#    - Useful for debugging distributed training logic locally
#    Example: NGPU=32 COMM_MODE="local_tensor" ./run_train.sh



NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}
COMM_MODE=${COMM_MODE:-""}

export NCCL_NVLS_ENABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_10,mlx5_11"


# for torch2.10 compatibility
cd /vast/sangwu/testing/torchtitan
source /vast/sangwu/envs/torch210/.venv/bin/activate
export LD_LIBRARY_PATH=/vast/sangwu/envs/torch210/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH


echo "Starting training with $WORLD_SIZE machines"
echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "RANK: $RANK"
echo "MODULE: $MODULE"
echo "CONFIG: $CONFIG"


if [ "$WORLD_SIZE" -gt 1 ]; then
    torchrun \
        --nproc_per_node=${NGPU} \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@"
else
     torchrun \
        --nproc_per_node=${NGPU} \
        --nnodes=1 \
        --standalone \
        -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@"
fi
