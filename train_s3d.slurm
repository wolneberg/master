#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G            
#SBATCH --job-name="s3d"
#SBATCH --output=s3d.out

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

# module purge
module load Python/3.10.4-GCCcore-11.3.0
# python -m venv .s3d
source .s3d/bin/activate

# module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
pip install -q torch torchvision torchaudio pandas matplotlib

python train_s3d.py

deactivate
