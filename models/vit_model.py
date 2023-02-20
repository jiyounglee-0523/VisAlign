import torch
from torch import nn
import torchvision
from torchvision.models import vit_b_16, vit_l_16, vit_h_14, vit_l_32
from torchvision.models import ViT_L_32_Weights
from torchvision.models.vision_transformer import VisionTransformer


class VisionTransformerModule(nn.Module):
    def __init__(self,
                 model_name,
                 num_classes,
                 pretrained_weights,
                 freeze_weights,
                 ):
        super().__init__()

        if pretrained_weights is True:
            self.model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)

            # change the last layer to match num_classes
            self.model.heads[0] = nn.Linear(1024, num_classes, bias=True)

            # option to freeze weights
            if freeze_weights is True:
                for name, param in self.model.named_parameters():
                    if name.split('.')[0] != 'heads':
                        param.requires_grad = False

        elif pretrained_weights is False:
            if model_name == 'vit_b_16':
                self.model = vit_b_16(num_classes=num_classes)

            elif model_name == 'vit_l_16':
                self.model = vit_l_16(num_classes=num_classes)

            elif model_name == 'vit_h_14':
                self.model = vit_h_14(num_classes=num_classes)

            elif model_name == 'vit_30_16':
                image_size = 224
                patch_size = 16
                num_layers = 30
                num_heads = 16
                hidden_dim = 1024
                mlp_dim = 4096

                self.model = VisionTransformer(
                    image_size=image_size,
                    patch_size=patch_size,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    num_classes=num_classes
                )

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Inputs:
            x - Tensor representing the image of shape [B, C, H, W]
            patch_size - Number of pixels per dimension of the patches (integer)
            flatten_channels - If True, the patches will be returned in a flattened format
                               as a feature vector instead of a image grid.
        """

        out = self.model(x)
        return out



'''
def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

class VisionTransformer(nn.Module):
    def __init__(self,
            patch_size,
            num_channels,
            num_heads,
            embed_dim,
            hidden_dim,
            num_layers,
            num_classes,
            dropout,
            num_patches,
            ):
        super().__init__()

        self.patch_size = patch_size


        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)


        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        cls = x[0]
        out = self.mlp_head(cls)
        return out
'''