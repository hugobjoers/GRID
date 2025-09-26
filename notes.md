# Inference format

The result from inference is a list of dictionaries of format dict_keys(['item_id', 'embedding']) where item_id=int and embedding is a 1024 (this corresponds to "d_model" in options.json of your model) long list of floats.