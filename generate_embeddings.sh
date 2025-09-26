#!/bin/bash
source "./options.conf"
python3.10 -m src.inference experiment=sem_embeds_inference_flat data_dir=$data_dir