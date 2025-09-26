#!/bin/bash
source "./options.conf"
new_num=$((num_hierarchies + 1))
python3.10 -m src.inference experiment=tiger_inference_flat data_dir=$data_dir semantic_id_path=$sid_path ckpt_path=$model_path num_hierarchies=$new_num