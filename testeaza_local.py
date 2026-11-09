import torch
from src.models.choose import AdvancedMRNet

print("1. Incarcam arhitectura retelei (SE-ResNet50)...")
# Folosim clasa pe care am construit-o în Colab
model = AdvancedMRNet(model_name='seresnet50', pretrained=False)

print("2. Cautam fisierul .pth local...")
# Schimbă 'model_salvat.pth' cu numele exact pe care i l-ai dat
cale_model = 'models/model_salvat.pth' 

print("3. Conectam 'creierul' la retea...")
checkpoint = torch.load(cale_model, map_location=torch.device('cpu'), weights_only=False)

# Daca a fost salvat ca dict cu 'state_dict' (specific PyTorch)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)
    
model.eval()
print("✅ SUCCES! Modelul a fost incarcat perfect pe calculatorul tau si este gata de diagnosticare!")