#!/bin/sh
nvidia-smi

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load Python/3.10.4-GCCcore-11.3.0
#module load cuDNN/8.8.0.121-CUDA-12.0.0
# python -m venv .s3d # for å lage virutal environment, trenger bare bruke en gang
# source .s3d/bin/activate

# module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
pip install -q -r req.txt

echo "not fine tuning all layers"

python train_s3d.py -n $NAME -e $EPOCHS -b $BATCH_SIZE -t $LAYERS
