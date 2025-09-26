#!/bin/bash
source "./options.conf"
python3.10 -m src.inference experiment=rkmeans_inference_flat data_dir=$data_dir embedding_path=$embedding_path embedding_dim=$embedding_dim num_hierarchies=$num_hierarchies codebook_width=$codebook_width ckpt_path=$sid_model_path