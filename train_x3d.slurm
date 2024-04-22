#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=14:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G  
#SBATCH --job-name="SLR"
#SBATCH --output=Models/x3d/Output/%x.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32g



#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=helenaml@stud.ntnu.no

#mem: total memory per node
#mem-per-cpu: memory per cpu-core

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

echo "--- nvidia-smi ---"
nvidia-smi

module purge
module list
module load Python/3.10.4-GCCcore-11.3.0
# # module load  Python/3.11.3-GCCcore-12.3.0
# # module load Clang/16.0.6-GCCcore-12.3.0 


pip install -q -r req.txt
pip install -U git+https://github.com/facebookresearch/fvcore.git
echo "x3d med gpu"
# python -v
python train_x3d.py -n movinet1

echo "Finished running train.py"

deactivate
# conda deactivate