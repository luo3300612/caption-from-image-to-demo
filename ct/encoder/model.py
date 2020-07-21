import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18 as resnet


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = nn.Sequential(*list(resnet(pretrained=False).children())[:-1])
        self.out = nn.Linear(512, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.out(x)
        return x


if __name__ == '__main__':
    net = Encoder()
    print(net)
    x = torch.randn(16,3,256,256)
    res = net(x)
    print(res.shape)
