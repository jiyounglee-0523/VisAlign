import torch
from torch import Tensor, nn
import torch.nn.functional as F
from copy import deepcopy
from flash.core.optimizers import LARS
import pl_bolts

import pytorch_lightning as pl
from pl_model_selfsupervised import SSLBaseModule
from models.vit_model import VisionTransformerModule
from models.convnext_model import ConvNext
from models.mlpmixer_model import MLPMixerModule
from models.swin_transformer_model import SwinTransformerModule
from models.densenet_model import DenseNetModule
    
def l2_norm(x, y):
    # L2 normalization
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)

class BYOLModule(SSLBaseModule):
    def __init__(self, args):
        super().__init__(args)

        assert isinstance(self.model, (VisionTransformerModule, DenseNetModule, ConvNext, MLPMixerModule, SwinTransformerModule))


        # hyperparams
        hidden_size = 4096
        proj_size = 256
        self.decay_rate = 0.996

        self.projection_head = nn.Sequential(
            nn.Linear(self.model.hidden_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, proj_size)
        )


        self.target_model = deepcopy(self.model)
        self.target_projection_head = deepcopy(self.projection_head)

        for params_target in self.target_model.parameters():
            params_target.requires_grad = False
        for params_target in self.target_projection_head.parameters():
            params_target.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(proj_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, proj_size)
        )

        self.save_hyperparameters()

    def forward(self, x):
        x1, x2, _ = x
        logits_1, logits_2 = self.predictor(self.projection_head(self.model(x1))), self.predictor(self.projection_head(self.model(x2)))

        with torch.no_grad():
            targ_1, targ_2 = self.target_projection_head(self.target_model(x1)), self.target_projection_head(self.target_model(x2))
            
        return logits_1, logits_2, targ_1, targ_2
        # return self.model(x)
    
    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.target_model.parameters()):
            param_k.data = param_k.data * self.decay_rate + param_q.data * (1. - self.decay_rate)

        for param_q, param_k in zip(self.projection_head.parameters(), self.target_projection_head.parameters()):
            param_k.data = param_k.data * self.decay_rate + param_q.data * (1. - self.decay_rate)

    def _calculate_loss(self, batch, mode='train'):

        logits_1, logits_2, targ_1, targ_2 = self(batch)

        loss = l2_norm(logits_1, targ_2)
        loss += l2_norm(logits_2, targ_1)

        loss = loss.mean()

        self.log(mode + "_loss", loss)
        
        # with torch.no_grad():
        #     x1, x2, _ = batch
        #     x = torch.concat((x1, x2))

        #     proj_feats = self.predictor(self.projection_head(self.model(x)))

        #     pairwise_sim = F.cosine_similarity(proj_feats[:, None, :], proj_feats[None, :, :], dim=-1)
            
        #     self_mask = torch.eye(pairwise_sim.shape[0], dtype=torch.bool, device=pairwise_sim.device)
        #     pairwise_sim.masked_fill_(self_mask, -9e15)

        #     pos_mask = self_mask.roll(shifts=pairwise_sim.shape[0] // 2, dims=0)

        #     # Get ranking position of positive example
        #     comb_sim = torch.cat(
        #         [pairwise_sim[pos_mask][:, None], pairwise_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
        #         dim=-1,
        #     )
        #     sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        #     # Logging ranking metrics
        #     self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), sync_dist=True)
        #     self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean(), sync_dist=True)
        #     self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean(), sync_dist=True)

        return loss


    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, 'train')
        self._update_target_network_parameters()
        return loss

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, 'val')

    def configure_optimizers(self):
        # optimizer
        optimizer = LARS(
            list(self.model.parameters()) + list(self.projection_head.parameters()) + list(self.predictor.parameters()),
            lr = 0.2 * (self.args.batch_size * self.args.n_gpus / 256),
            weight_decay=1.5e-6
        )

        # lr scheduler
        scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=10,
            max_epochs=1000,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }