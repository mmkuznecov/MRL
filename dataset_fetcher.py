from datasets import load_dataset

# Download the train split
ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")
ds.save_to_disk("data/imagenet_1k_resized_256")

# You can also download the validation split
ds_val = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")
ds_val.save_to_disk("data/imagenet_1k_resized_256_val")

ds_val = load_dataset("evanarlian/imagenet_1k_resized_256", split="test")
ds_val.save_to_disk("data/imagenet_1k_resized_256_test")