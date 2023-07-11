![VisAlign Overview](figures/DatasetOverview.png)

# VisAlign: Dataset for Measuring the Degree of Alignment between AI and Humans in Visual Perception
By
Jiyoung Lee,
Seungho Kim,
Seunghyun Won,
Joonseok Lee,
Marzyeh Ghassemi,
James Thorne,
Jaeseok Choi,
O-Kil Kwon,
Edward Choi


## Requirements
- `pytorch==1.12.1`
- `pytorch-lightning==1.8.5.post0`
- `lightning-bolts==0.6.0.post1`
- `lightning-flash==0.8.1.post0`
- `wandb==0.15.3`
- `scikit-image==0.20.0`
- `timm==0.9.2`
- `mlp-mixer-pytorch==0.1.1`

## Dataset
The train set and the open test set can be downloaded from [here](https://www.dropbox.com/scl/fi/e5p5epgvg2d9bniy1v81e/VisAlign.tar.gz?rlkey=69mhbl4v5uowpy27ox6si9qgo&dl=0).

After extracting the file, you will have the following files/directories:
```
open_test_corruption_labels.pk 
open_test_set/
train_files/
train_split_filenames/
├─  final_eval/
└─  final_train/
```
In the `config/imagenet.yaml` file, replace the following paths:
```
...
dataset:
  ...
  train:
    label_path: {path to train_split_filenames/final_train}
    imagenet21k_path: {path to train_files}
    ...
  eval:
    label_path: {path to train_split_filenames/final_eval}
    imagenet21k_path: {path to train_files}
    ...
```

## Train
You can train a baseline model using the following command:
```
python main.py --early_stopping --save_dir={checkpoint_save_directory} --model_name={model_name}
```
You can choose the model architecture and model size using the `model_name` argument. The model sizes we used in our baseline experiments are as follows:
- ViT: `vit_30_16`
- Swin Transformer: `swin_extra`
- ConvNeXt: `convnext_extra`
- DenseNet: `densenet_extra`
- MLP-Mixer: `mlp`

To finetune pre-trained model, please change `pretrained_weights` and `freeze_weights` to `True`.

Also, you can train Self-supervised model using the following command:
```
python main.py --early_stopping --save_dir={checkpoint_save_directory} --model_name={model_name} --ssl --ssl_type={ssl_type}
```
You can choose the ssl type using `ssl_type` argument.
- SimCLR: `simclr`
- BYOL: `byol`
- DINO: `dino`


## Evaluate
You can evaluate abstention function using the following command:
```
python test_main.py 
  --save_dir {save_dir}                       # directory to save abstention function result
  --ckpt_dir {ckpt_dir}                       # directory where model checkpoints exist
  --model_name {model_name}                   # model name we want to evaluate
  --postprocessor_name {abstention_function}  # name of the postprocessor
  --test_dataset_path {test_dataset_path}     # path to open_test_set
  --train_dataset_path {train_dataset_path}   # path to train set, this is needed to calculate distance for distance-based functions
```
You can choose the abstention function using `postprocessor_name` argument. The choices of abstention functions are `knn`, `mcdropout`, `mds`, `odin`, `msp`, `tapudd`.

You can evaluate a model's visual alignment via Hellinger's distance as described in our paper.
<!-- This implementation additionally allows you to report the proposed reliability score, which lets you choose a cost value *c* for incorrect decisions. -->
```
python evaluate_visual_alignment.py 
  --save_dir {save_dir}                 # directory where the absention function results are stored
  --test_filenames_path {dataset_path}  # directory where test dataset filenames are stored
  --corruption_path                     # open_test_corruption_labels.pk file path
```

## Citation
```
This section will be updated upon acceptance.
```