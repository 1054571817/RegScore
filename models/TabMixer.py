"""
based on https://github.com/SanoScience/TabMixer/
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from torch import nn


class TabularModule(nn.Module):
    def __init__(self, tab_dim=6,
                 channel_dim=2,
                 frame_dim=None,
                 hw_size=None,
                 module=None):
        """

        Args:
            tab_dim: Number of tabular features
            channel_dim: Number of channels
            frame_dim: Number of frames
            hw_size: Spatial dimensions (height and width)
            module: Module name (optional)
        """
        super(TabularModule, self).__init__()
        self.channel_dim = channel_dim
        self.tab_dim = tab_dim
        self.frame_dim = frame_dim
        self.hw_size = hw_size


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)


class TabMixer_block(TabularModule):
    def __init__(self,
                 norm=Affine,
                 **kwargs
                 ):
        """
        TabMixer module for integrating tabular data in vision models via mixing operations.

        Args:
            norm: Normalization layer
            use_tabular_data: Do not use tabular data for mixing operations
            spatial_first:  Apply spatial mixing as first mixing operation
            use_spatial: Apply spatial mixing
            use_temporal: Apply temporal mixing
            use_channel: Apply channel mixing
            **kwargs:
        """
        super(TabMixer_block, self).__init__(**kwargs)

        self.tab_embedding = MLP(in_features=self.tab_dim, hidden_features=self.tab_dim // 2,
                                 out_features=self.tab_dim)

        self.tab_mlp_s = MLP(in_features=(self.hw_size[0] * self.hw_size[1]) // 4 + self.tab_dim,
                             hidden_features=(self.hw_size[0] * self.hw_size[1]) // 8,
                             out_features=self.hw_size[0] * self.hw_size[1] // 4)
        self.norm_s = norm((self.hw_size[0] * self.hw_size[1]) // 4)

        self.tab_mlp_c = MLP(in_features=self.channel_dim + self.tab_dim, hidden_features=self.channel_dim // 2,
                             out_features=self.channel_dim)
        self.norm_c = norm(self.channel_dim)

        self.pool = nn.AvgPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x, tab=None):
        x = x.unsqueeze(2)
        B, C_in, F_in, H_in, W_in = x.shape
        x = self.pool(x)
        _, C, F, H, W = x.shape

        x = x.reshape(B, C, F, (H * W))

        tab = torch.unsqueeze(tab, 1).unsqueeze(1)
        tab_emb = self.tab_embedding(tab)

        tab_emb_spatial = tab_emb.repeat(1, C, F, 1)
        tab_emb_channel = tab_emb.repeat(1, H * W, F, 1)

        x_s = self.norm_s(x)
        x_s = torch.concat([x_s, tab_emb_spatial], dim=3)
        x_s = self.tab_mlp_s(x_s)
        x = x + x_s
        x = rearrange(x, 'b c f hw -> b hw f c')

        x_c = self.norm_c(x)
        x_c = torch.concat([x_c, tab_emb_channel], dim=3)
        x_c = self.tab_mlp_c(x_c)
        x = x + x_c
        x = rearrange(x, 'b hw f c -> b c f hw')

        x = x.reshape(B, C, F, H, W)
        x = nn.functional.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear')
        return x.squeeze(2)


class TabMixer(nn.Module):
    """
    Evaluation model for imaging and tabular data.
    """

    def __init__(self, args) -> None:
        super(TabMixer, self).__init__()
        self.args = args
        self.imaging_encoder = torchvision_ssl_encoder(args.model, return_all_feature_maps=True)
        self.imaging_encoder.layer4 = torch.nn.Sequential(*list(self.imaging_encoder.layer4.children())[:-1])
        self.tabular_encoder = nn.Identity()
        self.tabmixer = TabMixer_block(channel_dim=args.embedding_dim, tab_dim=args.input_size, hw_size=(4, 4))

        in_ch, out_ch = args.embedding_dim // 4, args.embedding_dim
        self.residual = nn.Sequential(nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(in_ch),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(in_ch),
                                      nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_ch)
                                      )
        self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True)
        in_dim = args.embedding_dim
        self.head = nn.Linear(in_dim, args.num_classes)

        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module, init_gain=0.02) -> None:
        """
        Initializes weights according to desired strategy
        """
        if isinstance(m, nn.Linear):
            if self.args.init_strat == 'normal':
                nn.init.normal_(m.weight.data, 0, 0.001)
            elif self.args.init_strat == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif self.args.init_strat == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.args.init_strat == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_im = self.imaging_encoder(x[0])[-1]
        x = self.tabmixer(x_im, x[1])
        x = self.residual(x)
        x = x + self.shortcut(x_im)
        x = self.act(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = TabMixer_block(channel_dim=128, tab_dim=90, hw_size=(8, 8))
    img = torch.randn(4, 128, 8, 8)
    tab = torch.randn(4, 90)
    y = model(img, tab)
    print(y.shape)
