import torch
from torch import nn
import torchvision
from torchvision.models import swin_t, swin_s, swin_b
from torchvision.models.swin_transformer import SwinTransformer


class SwinTransformerModule(nn.Module):
    def __init__(self,
                 model_name,
                 num_classes
                 ):
        super().__init__()

        if model_name == 'swin_t':
            self.model = swin_t(num_classes=num_classes)

        elif model_name == 'swin_s':
            self.model = swin_s(num_classes=num_classes)

        elif model_name == 'swin_b':
            self.model = swin_b(num_classes=num_classes)

        elif model_name == 'swin_extra':
            patch_size = [4, 4]
            embed_dim = 256
            depths = [2, 2, 15, 2]
            num_heads = [4, 8, 16, 32]
            window_size = [7, 7]
            stochastic_depth_prob = 0.5

            self.model = SwinTransformer(
                patch_size=patch_size,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                stochastic_depth_prob=stochastic_depth_prob,
                num_classes=num_classes
            )

        else:
            raise NotImplementedError

    def forward(self, x):
        '''
        x embedding model
                layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )
        '''

        out = self.model(x)
        return out