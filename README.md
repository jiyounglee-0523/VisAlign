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
The train set and the open test set can be downloaded from [here](https://www.dropbox.com/scl/fi/s2ncdwnz5uk9jk1kb6rg9/VisAlign_dataset.tar.gz?rlkey=kkgvorfp6893xrdddktfhzf7w&dl=0).

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
python main.py
  --config {config}                                   # path to the config yaml file
  --seed {seed}                                       # environment seed
  --early_stopping                                    # activate early stopping
  --early_stopping_patience {early_stopping_patience} # number of epochs for early stopping
  --save_dir {save_dir}                               # path to save checkpoints
  --n_epochs {n_epochs}                               # number of epochs
  --save_top_k {save_top_k}                           # number of model checkpoints to save
  --reload_ckpt_dir {reload_ckpt_dir}                 # continue an unfinished session
  --n_gpus {n_gpus}                                   # number of GPUs to use during training
  --model_name {model_name}                           # name of the model
  --batch_size {batch_size}                           # batch size
  --ssl                                               # option to train self-supervised
  --ssl_type {ssl_type}                               # self-supervised learning method
  --cont_ssl                                          # option to fine-tune an SSL-trained model
  --ssl_ckpt_dir {ssl_ckpt_dir}                       # path to saved SSL-trained model checkpoint
```
You can choose the model architecture and model size using the `model_name` argument. The model sizes we used in our baseline experiments are as follows:
- ViT: `vit_30_16`
- Swin Transformer: `swin_extra`
- ConvNeXt: `convnext_extra`
- DenseNet: `densenet_extra`
- MLP-Mixer: `mlp`

You can choose from the following SSL methods for the `ssl_type` argument.
- SimCLR: `simclr`
- BYOL: `byol`
- DINO: `dino`

To finetune pre-trained model, please change `pretrained_weights` and `freeze_weights` to `True` in `config/imagenet.yaml`.

To get started, Here are simple commands for training and SSL training:
```
# simple command for training
python main.py --early_stopping --save_dir {checkpoint_save_directory} --model_name {model_name}

# simple command for SSL training
python main.py --early_stopping --save_dir {checkpoint_save_directory} --model_name {model_name} --ssl --ssl_type {ssl_type}
```


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
  --seed {seed}                               # seed used when training, used for locating result filename
```
You can choose the abstention function using `postprocessor_name` argument. The choices of abstention functions are `knn`, `mcdropout`, `mds`, `odin`, `msp`, `tapudd`.

You can evaluate a model's visual alignment via Hellinger's distance as described in our paper.
<!-- This implementation additionally allows you to report the proposed reliability score, which lets you choose a cost value *c* for incorrect decisions. -->
```
python evaluate_visual_alignment.py 
  --save_dir {save_dir}                 # directory where the absention function results are stored
  --test_filenames_path {dataset_path}  # directory where test dataset filenames are stored
  --corruption_path                     # open_test_corruption_labels.pk file path
  --seed {seed}                         # seed used when training
  --model_name {model_name}             # model name we want to evaluate
  --ood_method {ood_method}             # name of the postprocessor
```

## Citation
```
This section will be updated upon acceptance.
```
