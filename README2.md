To download `imagenet_1k_resized_256` run `dataset_fetcher.py`

To make optimized version of dataset run `convert_hf_to_ffcv.py`

Example of usage below:

```bash
python convert_hf_to_ffcv.py \
    --dataset "evanarlian/imagenet_1k_resized_256" \
    --output "data/imagenet_train.dat" \
    --split "train" \
    --max-resolution 256 \
    --jpeg-quality 90 \
    --num-workers 8
```

To run train example to check if everything works okay: 

```bash
cd train/
python train_custom.py --config train_config_custom.yaml
```

In `tests.ipynb` there is example of loading weights.