import timm
import torch
import torch.nn as nn
from typing import Optional
from tactilebasedperception.models.autoencoder.encoder import EncoderCNN

class FeatureExtractor(nn.Module):

    DINO_HUB="facebookresearch/dino:main"
    DINOV2_HUB ="facebookresearch/dinov2"
    DINO_MODELS=['dino_vits16','dino_vits8','dino_vitb16','dino_vitb8',
                 'dino_xcit_small_12_p16','dino_xcit_small_12_p8',
                 'dino_xcit_medium_24_p16','dino_xcit_medium_24_p8','dino_resnet50']
    DINOV2_MODELS=['dinov2_vits14','dinov2_vitb14','dinov2_vitl14','dinov2_vitg14']

    ENCODER_CNN = "encodercnn"

    def __init__(self, 
                 name: str = 'dino_vitb8', 
                 features_key: Optional[str] = None,
                 custom_ckpt: Optional[str] = None):
        
        super().__init__()

        if name in self.DINO_MODELS:
            self.model = torch.hub.load(self.DINO_HUB, name)
        elif name in self.DINOV2_MODELS:
            self.model = torch.hub.load(self.DINOV2_HUB, name)
        elif name.lower() == ENCODER_CNN:
            self.model = EncoderCNN(image_size_w = 240, image_size_h = 320, latent_size = 128)
        else:
            self.model = timm.create_model(name, num_classes=0, pretrained=True)  

        if custom_ckpt:
            if name.lower() == ENCODER_CNN:
                self.model = torch.nn.DataParallel(self.encoder)
                self.model.load_state_dict(torch.load(custom_ckpt)['model_state_dict_encoder'])

            else:
                state_dict = torch.load(custom_ckpt, map_location='cpu')["model"]
                self.model.load_state_dict(state_dict, strict=False)

        self.features_key = features_key

    def forward(self, x):
        out = self.model(x)
        if self.features_key: out = out[self.features_key]
        return out