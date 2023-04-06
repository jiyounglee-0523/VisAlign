import torch
import torch.nn

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pl_model import BaseModule
from pl_model_selfsupervised import SSLBaseModule
from models_selfsupervised.simclr_model import SimCLRModule

from dataset.imagenet import ImageNetModule
from dataset.imagenet_pretrain import ImageNetPretrainModule
from dataset.imagenet_selfsupervised import ImageNetSSLModule

from models_selfsupervised.model_types import SIMCLR

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
            monitor='val_acc',
            patience=args.early_stopping_patience,
            mode='max',
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Save File Callback
    if not args.debug:
        if not args.ssl:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=args.save_dir,
                filename="{epoch:06}--{val_acc:.2f}",
                verbose=True,
                save_last=True,
                monitor='val_acc',
                save_top_k=args.save_top_k,
                mode='max',
            )
            callbacks.append(checkpoint_callback)
        else:
            if args.ssl_type == SIMCLR:
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=args.save_dir,
                    filename="{epoch:06}--{val_acc_top5:.2f}",
                    verbose=True,
                    save_last=True,
                    monitor='val_acc_top5',
                    save_top_k=args.save_top_k,
                    mode='max',
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
    parser.add_argument('--early_stopping_patience', default=15, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_dir', default='./', type=str)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--save_top_k', default=10, type=int)
    parser.add_argument('--reload_ckpt_dir', type=str)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--wandb_run_name', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--ssl', action='store_true')
    parser.add_argument('--ssl_type', type=str, choices=[SIMCLR])

    args = parser.parse_args()

    assert args.config is not None, 'Please specify config file'

    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        args.__setattr__(k, v)

    # change batch size argument
    args.dataset['batch_size'] = args.batch_size

    # Function for setting the seed
    pl.seed_everything(args.seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # logger
    if not args.debug:
        logger = pl.loggers.WandbLogger(config=args, project=args.wandb_run_name, entity='image-reliability', save_dir=f'/home/edlab/{os.getlogin()}/RELIABLE/reliable_project')

    

    # Call Dataset
    if args.ssl is True:
        assert args.ssl_type is not None
        dataloader = ImageNetSSLModule(args)

        # Call Model
        model = None

        if args.ssl_type == SIMCLR:
            model = SimCLRModule(args)

        assert model is not None

    else:
        if args.dataset['name'] == 'imagenet_pretrain':
            dataloader = ImageNetPretrainModule(args)
        else:
            dataloader = ImageNetModule(args)

        # Call Model
        model = BaseModule(args)

    # Call Trainer Config
    trainer_config = get_train_config(args)

    if args.debug:
        trainer = pl.Trainer(
            **trainer_config,
            num_sanity_val_steps=2,
            gradient_clip_val=0.5,
            accelerator='cuda',
            log_every_n_steps=10,
            strategy=DDPStrategy(find_unused_parameters=False),
            # plugins=DDPStrategy(find_unused_parameters=False),
        )
    elif not args.debug:
        trainer = pl.Trainer(
            **trainer_config,
            num_sanity_val_steps=3,
            gradient_clip_val=0.5,
            accelerator='cuda',
            log_every_n_steps=10,
            strategy=DDPStrategy(find_unused_parameters=False),
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
    # trainer.test()


if __name__ == '__main__':
    main()