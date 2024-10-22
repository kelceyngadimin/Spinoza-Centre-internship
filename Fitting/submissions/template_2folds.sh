#!/bin/bash
#SBATCH -t ---time---
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=---n---

#SBATCH -p ---p---
#SBATCH --output=/home/kngadimin/errout/%A_sub----subject---_ses----session---_slice----data_portion---.out
#SBATCH --error=/home/kngadimin/errout/%A_sub----subject---_ses----session---_slice----data_portion---.err

#SBATCH --mail-type=END
#SBATCH --mail-user=kelceyngadimin@gmail.com

module load 2023
source activate prffitting

cd /home/kngadimin/software/dnfitting/scripts
python prf_fitting_norm_snellius.py ---subject--- 128 ---data_portion--- ---session---

