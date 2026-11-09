import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class AdvancedMRNet(nn.Module):
    # Arhitectură MRNet modernizată cu SE-ResNet sau Vision Transformer
    def __init__(self, model_name='seresnet50', pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Clasificatorul final binar (bolnav/sănătos)
        self.classifier = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        
        if 'vit' in self.model_name or 'swin' in self.model_name:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        slice_features = self.backbone(x)
        patient_features, _ = torch.max(slice_features, dim=0, keepdim=True)
        return self.classifier(patient_features)

def get_model(name):
    print('[CHOOSE] Getting model:', name)
    if name in ['seresnet50', 'legacy_seresnet50', 'vit_base_patch16_224', 'vit_tiny_patch16_224']:
        return AdvancedMRNet(model_name=name, pretrained=True)
    else:
        raise ValueError('Modelul cerut nu este seresnet50 sau vit.')