#!/bin/bash -l

#$ -P ec520

# Request 1 CPUs
#$ -pe omp 1

# Request 1 GPU (the number of GPUs needed should be divided by the number of CPUs requested above)
#$ -l gpus=1

# Specify the minimum GPU compute capability
#$ -l gpu_c=4


#request 15 minutes maxium running time
#$ -l h_rt=1:00:00


module load anaconda3/4.4.0
source activate my_root
export CUDA_VISIBLE_DEVICES="1"
python main.py generate --wav_out_time 150
