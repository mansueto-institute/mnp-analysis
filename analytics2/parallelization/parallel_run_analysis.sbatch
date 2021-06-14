#!/bin/bash
#SBATCH --job-name=parallel_block_summary
#SBATCH --output=parallel_block_summary.out
#SBATCH --error=parallel_block_summary.err
#SBATCH --time=02:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-user=alexfeistritzer@uchicago.edu
#SBATCH --mail-type=ALL

module purge
module load parallel
module load python
ulimit -u 10000

srun="srun --exclusive -N1 -n1"
parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog runtask.log --resume"

base_path="/project2/bettencourt/mnp/prclz/data"
search_path="${base_path}/blocks/Africa/TZA"

#file_names=($(ls ${search_path}/*.csv | xargs -n 1 basename))
#num_files=$((${#file_names[@]} - 1)) #offset to be 0..n-1

#$parallel "$srun ./run_analysis.sbatch arg1:$base_path arg2:${file_names[{1}]} > run_analysis.sbatch.{1}" ::: $(seq 0 $num_files)

find $search_path -type f -name '*.csv' -print | xargs -n 1 basename | $parallel "$srun ./run_analysis.sbatch $base_path {}"