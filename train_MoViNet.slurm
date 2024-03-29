#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G            
#SBATCH --output=Models/MoViNet/Output/%x.out
#SBATCH --gres=gpu:1

#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ingrimwo@stud.ntnu.no

#mem: total memory per node
#mem-per-cpu: memory per cpu-core

#--gres=gpu:1

# -J "movinet-$(date +%Y%m%d-%H%M%S)"

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

# echo "--- nvidia-smi ---"
# nvidia-smi

module purge
# module load Anaconda3/2023.09-0
# module load Python/3.9.6-GCCcore-11.2.0
module load Python/3.10.4-GCCcore-11.3.0
# module load Python/3.10.8-GCCcore-12.2.0

# python -m venv env2
# source env2/bin/activate

module load cuDNN/8.8.0.121-CUDA-12.0.0
# module load cuDNN/8.4.1.50-CUDA-11.7.0 
# module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
# module load TensorFlow/2.13.0-foss-2023a
# module load TensorFlow/2.11.0-foss-2022a

pip install -q -r requirements.txt
pip install -q tf-models-official==2.14

pip install -q --upgrade tensorflow==2.14
pip install -q --upgrade tensorflow.text==2.14

# pip install tensorflow==2.13
# apt install --allow-change-held-packages libcudnn8=8.8.0.121-1+cuda12.0.0
# pip uninstall tf-nightly
# pip install --upgrade protobuf
# pip install -q git+https://github.com/tensorflow/docs

module list

echo "start train file"

python train.py

echo "Finished running train.py"

# deactivate

# python -m tf2onnx.convert --tflite Models/MoViNet/lite/model.tflite --output Models/MoViNet/lite/model.onnx --opset 13