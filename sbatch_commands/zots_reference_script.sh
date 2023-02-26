#! /bin/bash

# A name for the job
#SBATCH --job-name=train_inception
 
# The queue you want to run on. This will determine instance type
#SBATCH --partition=gpu

#Request GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=14G
#SBATCH --constraint=p32xlarge
 
# send email at start and end of job
#SBATCH --mail-type=END

#SBATCH --time=13-0
 
# to this email. must be NN email
#SBATCH --mail-user=zots@novonordisk.com
 
# Load the anaconda environment
module load anaconda3/2021.05
conda activate imagegpu

# and here is you actual job
python main.py