#!/bin/bash

# Define variables
model_class="lottery"
model="vgg16"
dataset_list=("cifar10")

# model_class="default"
# model="fc"
# dataset_list=("mnist")

experiment="singleshot"
pruner_list=("grasp")
# pruner_list=("synflow")
# post_epoch=100
# compression_list=(1)
compression_list=(0.05 0.1 0.2 0.5 2)
# compression=1

for dataset in "${dataset_list[@]}"; do
    for pruner in "${pruner_list[@]}"; do
        for compression in "${compression_list[@]}"; do

            # Set pre-epochs based on the pruner
            if [ "$pruner" == "mag" ]; then
                pre_epochs=200
            else
                pre_epochs=0
            fi
            # Adjust post_epoch based on the dataset
            if [ "$dataset" == "cifar10" ]; then
                post_epoch=100
            elif [ "$dataset" == "mnist" ]; then
                post_epoch=10
            fi

            # Create an experiment ID
            expid="${dataset}-${model}-${model_class}-${pruner}-preEpochs${pre_epochs}-postEpochs${post_epoch}-compression${compression}"

            # Run main.py with the specified arguments in the background
            nohup python -u main.py --model-class "$model_class" --model "$model" --dataset "$dataset" --experiment "$experiment" --pruner "$pruner" --compression "$compression" --post-epoch "$post_epoch" --pre-epochs "$pre_epochs" --expid "$expid" > "${expid}.out" 2>&1 &
        done
    done
done