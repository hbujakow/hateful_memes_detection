#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --partition=short
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikolaj.galkowski.stud@pw.edu.pl
#SBATCH --job-name=train_procap
#SBATCH --output=/home2/faculty/mgalkowski/logs/procap/train_12_11_2023_10_10.log

. /home2/faculty/mgalkowski/miniconda3/etc/profile.d/conda.sh
conda activate captions_procap

python main.py --SAVE_NUM 12112023 --SEED 1111 --DATASET 'mem' --LR_RATE 1.3e-05
# python main.py --SAVE_NUM 822 --SEED 1112 --DATASET 'mem' --LR_RATE 1.3e-05
# python main.py --SAVE_NUM 823 --SEED 1113 --DATASET 'mem' --LR_RATE 1.3e-05
# python main.py --SAVE_NUM 824 --SEED 1114 --DATASET 'mem' --LR_RATE 1.3e-05
# python main.py --SAVE_NUM 825 --SEED 1115 --DATASET 'mem' --LR_RATE 1.3e-05
# python main.py --SAVE_NUM 826 --SEED 1116 --DATASET 'mem' --LR_RATE 1.3e-05
# python main.py --SAVE_NUM 827 --SEED 1117 --DATASET 'mem' --LR_RATE 1.3e-05
# python main.py --SAVE_NUM 828 --SEED 1118 --DATASET 'mem' --LR_RATE 1.3e-05
# python main.py --SAVE_NUM 829 --SEED 1119 --DATASET 'mem' --LR_RATE 1.3e-05
# python main.py --SAVE_NUM 830 --SEED 1120 --DATASET 'mem' --LR_RATE 1.3e-05
