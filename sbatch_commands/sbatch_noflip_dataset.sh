#! /bin/bash

# A name for the job
#SBATCH --job-name=NoFlipDataset
 
# The queue you want to run on. This will determine instance type
#SBATCH --partition=compute

#SBATCH --constraint=c59xlarge
 
# send email at start and end of job
#SBATCH --mail-type=END

#SBATCH --time=2-0
 
# to this email. must be NN email
#SBATCH --mail-user=mopc@novonordisk.com

# writes errors to stderr file and output to stdout
# https://it.stonybrook.edu/help/kb/handling-job-output
#SBATCH -e errors_no_flip_log.txt
#SBATCH -o output_no_flip_log.txt
#SBATCH --open-mode=append
 
# Load the virtual environment
source env/bin/activate

# run datasets creation
python src/data/copy_noflip.py