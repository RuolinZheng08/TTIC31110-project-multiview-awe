#!/bin/bash
#SBATCH --mail-user=vzhao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/vzhao/slurm/slurm_out/%j.%N.stdout
#SBATCH --error=/home/vzhao/slurm/slurm_out/%j.%N.stderr
#SBATCH --workdir=/scratch/vzhao/ttic31110-agwe
#SBATCH --job-name=agwe_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --partition pascal
#SBATCH --gres=gpu:1
#SBATCH --mem=24000
#SBATCH --exclusive

pwd; hostname; date

./train.sh
