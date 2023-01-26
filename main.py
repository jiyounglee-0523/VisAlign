import torch
import torch.nn

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pl_model import BaseModule
from dataset.imagenet import ImageNetModule

import os
import argparse
import yaml

def get_train_config(args):
    callbacks = list()

    # Learning Rate
    lr_callback = pl.callbacks.LearningRateMonitor(
        logging_interval='step'
    )
    callbacks.append(lr_callback)

    # Resume from checkpoint


    # Early Stopping Callback
    if args.early_stopping is True:
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='eval_loss', # TODO: change!
            patience=args.early_stopping_patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Save File Callback
    if not args.debug:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.save_dir,
            filename="{epoch:06}--{eval_loss:.2f}",
            verbose=True,
            save_last=True,
            monitor='eval_loss',
            save_top_k=args.save_top_k,
            mode='min',
        )
        callbacks.append(checkpoint_callback)

    config = {
        'max_epochs': args.n_epochs,
        'gpus': args.n_gpus,
        'callbacks': callbacks
    }
    return config

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser()

    # utils
    parser.add_argument('--config', default='./config/imagenet.yaml', type=str)
    parser.add_argument('--seed', default=45)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--early_stopping_patience', default=30, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_dir', default='./', type=str)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--save_top_k', default=10, type=int)
    parser.add_argument('--reload_ckpt_dir', type=str)
    parser.add_argument('--n_gpus', default=1, type=int)

    args = parser.parse_args()

    assert args.config is not None, 'Please specify config file'

    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        args.__setattr__(k, v)

    # Function for setting the seed
    pl.seed_everything(args.seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # logger
    if not args.debug:
        logger = pl.loggers.WandbLogger(config=args, project='XX', entity='XX')

    

    # Call Dataset
    dataloader = 'XX'

    # Call Model
    model = BaseModule(args)

    # Call Trainer Config
    trainer_config = get_train_config(args)

    if args.debug:
        trainer = pl.Trainer(
            **trainer_config,
            num_sanity_val_steps=1,
            gradient_clip_val=0.5,
            accelerator='cuda',
            # plugins=DDPStrategy(find_unused_parameters=False),
        )
    elif not args.debug:
        trainer = pl.Trainer(
            **trainer_config,
            num_sanity_val_steps=1,
            gradient_clip_val=0.5,
            accelerator='cuda',
            # plugins=DDPStrategy(find_unused_parameters=False),
            logger=logger,
        )

    if args.reload_ckpt_dir is not None:
        print(f"The model will be resumed from ckpt at {args.reload_ckpt_dir}")

    trainer.fit(
        model=model,
        datamodule=dataloader,
        ckpt_path=args.reload_ckpt_dir
    )
    trainer.test()


if __name__ == '__main__':
    main()