#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
#BSUB -J ddpm
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -M 20GB
#BSUB -W 01:00
## BSUB -u
### -- send notification at start -- 
## BSUB -B 
### -- send notification at completion -- 
## BSUB -N 

#BSUB -o Output.out 
#BSUB -e Output.err 

cd /zhome/87/3/155549/ADML
source .venv/bin/activate
cd w3

python ddpm.py train --data mnist --model mnist.pt --samples mnist.png --device cuda --batch-size 64 --epochs 50 --load_pretrained True