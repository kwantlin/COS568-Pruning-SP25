#!/bin/bash -l

#SBATCH -A lips
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kw2960@cs.princeton.edu


cd /n/fs/klips/COS568-Pruning-SP25/

# Set up Conda
source /u/kw2960/.bashrc
echo "test"

conda activate cos568
echo "started env"

# model_class="lottery"
# model="vgg16"
# dataset_list=("cifar10")

model_class="default"
model="fc"
dataset_list=("mnist")

experiment="singleshot"
pruner_list=("rand" "mag" "snip" "grasp" "synflow")
# pruner_list=("synflow")
post_epoch=100
compression_list=(1)
# compression_list=(0.05 0.1 0.2 0.5 1 2)
# compression=1
echo "starting loop"

# srun python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 0.5

for dataset in "${dataset_list[@]}"; do
    for pruner in "${pruner_list[@]}"; do
        for compression in "${compression_list[@]}"; do
            # Create an experiment ID
            expid="${dataset}-${model}-${model_class}-${pruner}-postEpochs${post_epoch}-compression${compression}"

            # Run main.py with the specified arguments in the background
            echo "running ${dataset} ${pruner} ${compression}"
            nohup main.py --model-class "$model_class" --model "$model" --dataset "$dataset" --experiment "$experiment" --pruner "$pruner" --compression "$compression" --post-epoch "$post_epoch" --expid "$expid" > "${expid}.out" &
        done
    done
done
