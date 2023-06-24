#!/usr/bin/bash

python ./src/pretext_task.py \
                    --device cuda:0 \
                    --num_epochs 2 \
                    --num_workers 4 
                    
python ./src/pretext_task.py \
                    --device cuda:0 \
                    --num_epochs 2 \
                    --num_workers 8 

python ./src/pretext_task.py \
                    --device cuda:0 \
                    --num_epochs 2 \
                    --num_workers 16