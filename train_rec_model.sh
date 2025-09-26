#!/bin/bash
source "./options.conf"
new_num=$((num_hierarchies + 1))
python -m src.train experiment=tiger_train_flat data_dir=$data_dir semantic_id_path=$sid_path num_hierarchies=$new_num