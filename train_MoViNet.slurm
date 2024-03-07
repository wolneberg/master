#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G            
#SBATCH --job-name="SLR"
#SBATCH --output=Models/MoViNet/Output/movinet_freez10_2.out
#SBATCH --gres=gpu:1



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
module load Python/3.10.4-GCCcore-11.3.0
# module load  Python/3.11.3-GCCcore-12.3.0

module load cuDNN/8.8.0.121-CUDA-12.0.0
# module load cuDNN/8.4.1.50-CUDA-11.7.0 
# module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load TensorFlow/2.13.0-foss-2023a

# source .gpu2/bin/activate

# module load Anaconda3/2023.09-0
# conda create --name .idun
# conda activate .idun

# module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
# module load cuDNN/8.8.0.121-CUDA-12.0.0

pip install -q -r requirements.txt
# pip install seaborn
# pip install tf2onnx
pip install -q tf-models-official
pip install -q git+https://github.com/tensorflow/docs

echo "movinet med gpu"
echo "wlasl-100: rmsprop, epochs=30, not freezed backbone, med shuffle av train dataset, over"

python train.py

echo "Finished running train.py"

# deactivate
# conda deactivate