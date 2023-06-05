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
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
    
def l2_norm(x, y):
    # L2 normalization
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)

def cancel_gradients_last_layer(epoch, model, frozen_epochs):
    if epoch >= frozen_epochs:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


class DINOModule(SSLBaseModule):
    def __init__(self, args):
        super().__init__(args)

        assert isinstance(self.model, (VisionTransformerModule, DenseNetModule, ConvNext, MLPMixerModule, SwinTransformerModule))


        # hyperparams
        hidden_size = 2048
        proj_size = 256
        dino_out_dim = 4096
        self.decay_rate = 0.996
        self.last_layer_frozen = 2

        self.teacher_temp = 0.04                          # teacher temperature
        self.student_temp = 0.1                           # student temperature
        self.center_momentum = 0.9                        # center momentum for EMA
        self.register_buffer("center", torch.zeros(1, dino_out_dim))

        self.projection_head = nn.Sequential(
            nn.Linear(in_features=self.model.hidden_dim, out_features=hidden_size),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=proj_size)
        )
            # if use_bn: layers.append(nn.BatchNorm1d(num_features=hidden_dim))
            # layers.append(nn.GELU() if use_gelu else nn.ReLU(inplace=True))
            
            # # adding all the other layers
            # for _ in range(num_layers-2):
            #     layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            #     if use_bn: layers.append(nn.BatchNorm1d(num_features=hidden_dim))
            #     layers.append(nn.GELU() if use_gelu else nn.ReLU(inplace=True))
                
            # layers.append(nn.Linear(in_features=hidden_dim, out_features=proj_dim))
            # layers.append(nn.Dropout(drop_p))

        self.last_layer = nn.utils.weight_norm(nn.Linear(in_features=proj_size, out_features=dino_out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

        self.teacher_last_layer = nn.utils.weight_norm(nn.Linear(in_features=proj_size, out_features=dino_out_dim, bias=False))
        self.teacher_last_layer.weight_g.data.fill_(1)
        self.teacher_last_layer.weight_g.requires_grad = False

        self.teacher_model = deepcopy(self.model)
        self.teacher_projection_head = deepcopy(self.projection_head)

        for params_teacher in self.teacher_model.parameters():
            params_teacher.requires_grad = False
        for params_teacher in self.teacher_projection_head.parameters():
            params_teacher.requires_grad = False
        for params_teacher in self.teacher_last_layer.parameters():
            params_teacher.requires_grad = False



        self.save_hyperparameters()
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output

        Args:
            teacher_output (List[torch.Tensor]): _description_
        """
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True) # (1, out_dim)
        self.center = self.center * self.center_momentum + (1 - self.center_momentum) * batch_center
    
    def student_forward(self, x):
        return self.last_layer(self.projection_head(self.model(x)))
    
    def teacher_forward(self, x):
        return self.teacher_last_layer(self.teacher_projection_head(self.teacher_model(x)))

    def forward(self, x):
        
        x_global = x[:2]
        x_local = x[2:]
        
        # Teacher Output - global crops
        teacher_crops = len(x_global)
        x_teacher = torch.cat(x_global, dim=0) # (batch_size * 2, 3, size, size)
        teacher_logits = self.teacher_forward(x_teacher) # (batch_size * 2, proj_dim)
        
        # Student Output - local + global crops
        # global + local
        student_crops = len(x)
        x_global_student = torch.cat(x_global, dim=0)
        x_local_student = torch.cat(x_local, dim=0)
        student_global_logits = self.student_forward(x_global_student)
        student_local_logits = self.student_forward(x_local_student)
        student_logits = torch.cat((student_global_logits, student_local_logits), dim=0) # (batch_size * n_crops, out_dim) -- n_crops is 2+n_local_crops (2 is global)
        
        return student_logits.chunk(student_crops), teacher_logits.chunk(teacher_crops)
    
    @torch.no_grad()
    def _update_teacher_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.teacher_model.parameters()):
            param_k.data = param_k.data * self.decay_rate + param_q.data * (1. - self.decay_rate)

        for param_q, param_k in zip(self.projection_head.parameters(), self.teacher_projection_head.parameters()):
            param_k.data = param_k.data * self.decay_rate + param_q.data * (1. - self.decay_rate)

        for param_q, param_k in zip(self.last_layer.parameters(), self.teacher_last_layer.parameters()):
            param_k.data = param_k.data * self.decay_rate + param_q.data * (1. - self.decay_rate)

    def _calculate_loss(self, output, mode='train'):
        """DINO loss computation

        Args:
            output (Tuple[torch.Tensor, torch.Tensor]): student_logits, teacher_logits

        Returns:
            torch.Tensor: DINO loss value
        """
        student_logits, teacher_logits = output # output already chunked in n_crops (global+local and global)
        student_out = [s / self.student_temp for s in student_logits]
        # teacher centering and sharpening
        teacher_out = [(t - self.center) / self.teacher_temp for t in teacher_logits]
        
        student_sm = [F.log_softmax(s, dim=-1) for s in student_out]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_out]
        
        total_loss = 0
        n_loss_terms = 0
        
        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue
                loss = torch.sum(-t*s, dim=-1) # (n_samples, )
                total_loss += loss.mean() # scalar
                n_loss_terms += 1
        total_loss /= n_loss_terms
        # update center param for teacher output
        self.update_center(teacher_logits)
        return total_loss
        
    def training_step(self, batch, batch_idx):
        
        x, views, _ = batch
        outputs = self(views)
        loss = self._calculate_loss(outputs)
        
        cancel_gradients_last_layer(
            epoch=self.current_epoch,
            model=self.model,
            frozen_epochs=self.last_layer_frozen
        )
        # EMA update        
        self._update_teacher_network_parameters()
        
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)
                
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, views, _ = batch
        outputs = self(views)
        loss = self._calculate_loss(outputs)
        
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss, sync_dist=True, prog_bar=True)
        
    def training_epoch_end(self, outputs):
        pass
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.projection_head.parameters()) + list(self.last_layer.parameters()),
            lr=0.2 * (self.args.batch_size * self.args.n_gpus / 256),
            weight_decay=1.5e-6
        )
        # lr scheduler
        self.lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=self.optimizer,
            warmup_epochs=10,
            max_epochs=20,
            warmup_start_lr=0.0,
            eta_min=0.000001,
            last_epoch=-1,
        )
        # else:
        return [self.optimizer], [self.lr_scheduler]


    # def training_step(self, batch, batch_idx):
    #     loss = self._calculate_loss(batch, 'train')
    #     self._update_teacher_network_parameters()
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     return self._calculate_loss(batch, 'val')

    # def configure_optimizers(self):
    #     # optimizer
    #     optimizer = LARS(
    #         list(self.model.parameters()) + list(self.projection_head.parameters()) + list(self.predictor.parameters()),
    #         lr = 0.2 * (self.args.batch_size * self.args.n_gpus / 256),
    #         weight_decay=1.5e-6
    #     )

    #     # lr scheduler
    #     scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
    #         optimizer,
    #         warmup_epochs=10,
    #         max_epochs=1000,
    #     )

    #     return {
    #         'optimizer': optimizer,
    #         'lr_scheduler': scheduler,
    #         'monitor': 'train_loss'
    #     }