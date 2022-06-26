from torch import nn
import torch
from torchvision import models

class KPDetector(nn.Module):
    """
    Predict K*N keypoints.
    """

    def __init__(self, num_tps, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps
        self.num_kps = kwargs['num_kps']

        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(num_features, num_tps * self.num_kps * 2) ## 直接坐标回归出若干个点

        
    def forward(self, image):

        fg_kp = self.fg_encoder(image)
        bs, _, = fg_kp.shape
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1
        out = {'fg_kp': fg_kp.view(bs, self.num_tps * self.num_kps, -1)} ## bs, self.num_tps*5, 2

        return out
