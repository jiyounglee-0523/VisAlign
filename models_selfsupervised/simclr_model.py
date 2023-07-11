import torch
from torch import nn
import torch.nn.functional as F
from flash.core.optimizers import LARS
import pl_bolts


from pl_model_selfsupervised import SSLBaseModule
from models.vit_model import VisionTransformerModule
from models.convnext_model import ConvNext
from models.mlpmixer_model import MLPMixerModule
from models.swin_transformer_model import SwinTransformerModule
from models.densenet_model import DenseNetModule

class SimCLRModule(SSLBaseModule):
    def __init__(self, args):
        super().__init__(args)

        self.projection_head = None

        self.temperature = 0.5

        if isinstance(self.model, (VisionTransformerModule, DenseNetModule, ConvNext, MLPMixerModule, SwinTransformerModule)):
            self.projection_head = nn.Sequential(
                nn.Linear(self.model.hidden_dim, 4 * self.model.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(4 * self.model.hidden_dim, self.model.hidden_dim)
            )

        assert self.projection_head is not None

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)


    def _calculate_loss(self, batch, mode='train'):
        imgs1, imgs2, _ = batch
        imgs = torch.concat((imgs1, imgs2))

        feats = self.model(imgs)

        proj_feats = self.projection_head(feats)

        pairwise_sim = F.cosine_similarity(proj_feats[:, None, :], proj_feats[None, :, :], dim=-1)
        
        self_mask = torch.eye(pairwise_sim.shape[0], dtype=torch.bool, device=pairwise_sim.device)
        pairwise_sim.masked_fill_(self_mask, -9e15)

        pos_mask = self_mask.roll(shifts=pairwise_sim.shape[0] // 2, dims=0)

        pairwise_sim = pairwise_sim / self.temperature
        nll = -pairwise_sim[pos_mask] + torch.logsumexp(pairwise_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [pairwise_sim[pos_mask][:, None], pairwise_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), sync_dist=True)
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean(), sync_dist=True)
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean(), sync_dist=True)

        return nll



    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, 'val')


    def configure_optimizers(self):
        # optimizer
        optimizer = LARS(
            list(self.model.parameters()) + list(self.projection_head.parameters()),
            lr = 0.3 * (self.args.batch_size * self.args.n_gpus / 256),
            weight_decay=1.5e-6
        )

        # lr scheduler
        scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=10,
            max_epochs=100,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }