#!/bin/bash

USER=$(whoami)

DATE=$(date +%Y%m%d-%H%M%S)

JOB_NAME="${1}-${DATE}"
EPOCHS=$2
BATCH_SIZE=$3
LAYERS=$4
NUM_CLASSES=$5

CODE_PATH="Models/S3D/"
OUTPUT_FILE="${CODE_PATH}/s3d_outputs/${JOB_NAME}.out"

echo "Current user is: $USER"
echo "Job name: $JOB_NAME"
echo "Code path: $CODE_PATH"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Trainable layers: $LAYERS"
echo "Num classes: $NUM_CLASSES"

echo "Running slurm job"
sbatch --partition=GPUQ \
  --account=ie-idi \
  --time=23:00:00 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --mem=10G \
  --gres=gpu:1 \
  --job-name=$JOB_NAME \
  --output=$OUTPUT_FILE \
  --export=USER=$USER,CODE_PATH=$CODE_PATH,NAME=$JOB_NAME,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE,LAYERS=$LAYERS,NUM_CLASSES=$NUM_CLASSES \
  --mail-type=end \
  --mail-type=fail \
  --mail-user=ingrimwo@stud.ntnu.no \
  train_s3d.slurm