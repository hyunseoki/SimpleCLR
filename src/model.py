import timm
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 head_type = 'nonlinear'):
        super().__init__()

        assert head_type in ['linear', 'nonlinear']

        if head_type == 'linear':
            self.layers = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        elif head_type =='nonlinear':
            self.layers = nn.Sequential(
                nn.Linear(in_channels, hidden_channels, bias=True),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )

    def forward(self,x):
        x = self.layers(x)
        return x


class PreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            model_name='resnet50',
            pretrained=False,
            in_chans=3,
            num_classes=0,
        )

        self.head = ProjectionHead(2048, 2048, 128)

    def forward(self,x):
        out = self.encoder(x)
        out = self.head(out)
        return out


if __name__ == '__main__':
    import torch
    model = PreModel()
    model.eval()
    x = torch.randn((3, 3, 28, 28))
    print(model(x).shape)