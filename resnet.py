from fastai.vision import create_cnn, create_body, create_head
from fastai.callbacks.hooks import num_features_model
import torchvision.models as models
import torch.nn as nn

def resnet(classes):
    base_model = models.resnet50(pretrained=True)
    body = nn.Sequential(*list(base_model.children())[:-2])
    nf = num_features_model(body) * 2
    head = create_head(nf, classes, None, ps=0.5, bn_final=False)
    return nn.Sequential(body, head, nn.LogSoftmax())
