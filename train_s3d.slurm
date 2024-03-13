#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G            
#SBATCH --output=Models/S3D/s3d_outputs/%x.out
#SBATCH --gres=gpu:1

#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ingrimwo@stud.ntnu.no


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

# for creating a video from a tensor
pip -q install av

EPOCHS=30
BATCH_SIZE=4
LAYERS=4
NUM_CLASSES=100

echo "Job name: $SLURM_JOB_NAME"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Trainable layers: $LAYERS"
echo "Num classes: $NUM_CLASSES"

python train_s3d.py -n $SLURM_JOB_NAME -e $EPOCHS -b $BATCH_SIZE -t $LAYERS -c $NUM_CLASSES
