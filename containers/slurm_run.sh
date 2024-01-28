#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=Blood-simulation
#SBATCH --output=Blood-simulation.log
#SBATCH --mem=10G
#SBATCH --ntasks=5
#SBATCH --gpus=1

export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
singularity run --nv --nvccli -B /usr/share/glvnd blood_sim-build_latest.sif ..
