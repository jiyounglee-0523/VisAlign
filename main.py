import torch
import torch.nn

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pl_model import BaseModule
from pl_model_selfsupervised import SSLBaseModule
from models_selfsupervised.simclr_model import SimCLRModule
from models_selfsupervised.byol_model import BYOLModule
from models_selfsupervised.dino_model import DINOModule

from dataset.imagenet import ImageNetModule
from dataset.imagenet_pretrain import ImageNetPretrainModule
from dataset.imagenet_selfsupervised import ImageNetSSLModule

from models_selfsupervised.model_types import SIMCLR, BYOL, DINO

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

    if args.ssl:
        monitor = 'val_acc_top5'
    else:
        monitor = 'val_acc'

    # Resume from checkpoint


    # Early Stopping Callback
    if args.early_stopping is True:
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=monitor,
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
                filename="{epoch:06}--{val_acc:.4f}",
                verbose=True,
                save_last=True,
                monitor=monitor,
                save_top_k=args.save_top_k,
                mode='max',
            )
            callbacks.append(checkpoint_callback)
        else:
            if args.ssl_type == SIMCLR:
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=args.save_dir,
                    filename="{epoch:06}--{val_acc_top5:.4f}",
                    verbose=True,
                    save_last=True,
                    monitor=monitor,
                    save_top_k=args.save_top_k,
                    mode='max',
                )
                callbacks.append(checkpoint_callback)
            if args.ssl_type == BYOL:
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=args.save_dir,
                    filename="{epoch:06}--{val_loss:.4f}",
                    verbose=True,
                    save_last=True,
                    # monitor=monitor,
                    # save_top_k=args.save_top_k,
                    # mode='max',
                )
                callbacks.append(checkpoint_callback)
            if args.ssl_type == DINO:
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=args.save_dir,
                    filename="{epoch:06}--{val_loss:.4f}",
                    verbose=True,
                    save_last=True,
                    # monitor=monitor,
                    # save_top_k=args.save_top_k,
                    # mode='max',
                )
                callbacks.append(checkpoint_callback)

    config = {
        'max_epochs': args.n_epochs,
        'devices': args.n_gpus,
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
    parser.add_argument('--wandb_exec_name', type=str)
    parser.add_argument('--wandb_run_name', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--ssl', action='store_true')
    parser.add_argument('--ssl_type', type=str, choices=[SIMCLR, BYOL, DINO])
    parser.add_argument('--cont_ssl', action='store_true')
    parser.add_argument('--ssl_ckpt_dir', type=str)

    args = parser.parse_args()

    assert args.config is not None, 'Please specify config file'

    assert not (args.ssl and args.cont_ssl), 'Choose either ssl or cont_ssl, but not both'

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
        logger = pl.loggers.WandbLogger(config=args, project=args.wandb_run_name, name=args.wandb_exec_name, entity='image-reliability', save_dir=f'/home/edlab/{os.getlogin()}/RELIABLE/reliable_project')

    # Call Dataset
    if args.ssl is True:

        assert args.ssl_type is not None

        # Call Model
        model = None

        dataloader = ImageNetSSLModule(args)

        if args.ssl_type == SIMCLR:
            model = SimCLRModule(args)
        elif args.ssl_type == BYOL:
            model = BYOLModule(args)
        elif args.ssl_type == DINO:
            model = DINOModule(args)


        assert model is not None

    else:
        if args.dataset['name'] == 'imagenet_pretrain':
            dataloader = ImageNetPretrainModule(args)
        else:
            dataloader = ImageNetModule(args)

        if args.cont_ssl:
            assert args.ssl_ckpt_dir is not None
            model = BaseModule(args)
            model.load_from_checkpoint(args.ssl_ckpt_dir, strict=False)
            model.model.replace_ssl()

        # Call Model
        else:
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