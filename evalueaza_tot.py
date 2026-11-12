import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from src.models.choose import AdvancedMRNet

def evalueaza_modelul():
    cale_model = 'models/model_salvat.pth'
    cale_csv = 'dataset/valid_abnormal.csv'
    folder_rmn = 'dataset/valid_files/axial/'

    model = AdvancedMRNet(model_name='seresnet50', pretrained=False)
    
    checkpoint = torch.load(cale_model, map_location=torch.device('cpu'), weights_only=False)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model.eval()

    print(f"=> Citim baremul de corectare din: {cale_csv}")
    df = pd.read_csv(cale_csv, header=None, names=['id', 'label'])
    
    y_adevarat = [] 
    y_predictii_prob = [] 
    y_predictii_binar = [] 
    
    print("=> Incepem analizarea celor 119 pacienti. Te rog asteapta...")
    
    with torch.no_grad():
        for index, row in df.iterrows():
            # Eliminam caracterele in plus (precum '_') inainte de a face numarul
            id_curatat = str(row['id']).replace('_', '').strip()
            nume_fisier = f"{int(id_curatat):04d}.npy"
            cale_fisier = os.path.join(folder_rmn, nume_fisier)
            
            if not os.path.exists(cale_fisier):
                print(f"Fisierul {nume_fisier} lipseste din {folder_rmn}. Sarim peste.")
                continue
                
            rmn_array = np.load(cale_fisier)
            tensor_rmn = torch.FloatTensor(rmn_array)
            
            tensor_rmn = torch.stack([tensor_rmn, tensor_rmn, tensor_rmn], dim=1)
            
            tensor_rmn = tensor_rmn.unsqueeze(0)
            
            rezultat_brut = model(tensor_rmn)
            probabilitate = torch.sigmoid(rezultat_brut).item()
            
            label_curatat = int(str(row['label']).replace('_', '').strip())
            y_adevarat.append(label_curatat)
            
            y_predictii_prob.append(probabilitate)
            
            decizie = 1 if probabilitate >= 0.5 else 0   
            y_predictii_binar.append(decizie)
            
            if (index + 1) % 20 == 0:
                print(f"   Procesat: {index + 1} / {len(df)}")
    acuratete = accuracy_score(y_adevarat, y_predictii_binar) * 100
    auc = roc_auc_score(y_adevarat, y_predictii_prob)

    print("\n" + "="*40)
    print("REZULTATE FINALE TESTARE GLOBALA:")
    print("="*40)
    print(f"Total pacienti validati: {len(y_adevarat)}")
    print(f"Acuratete pura: {acuratete:.2f}%")
    print(f"Scor AUC: {auc:.4f}")
    print("="*40)

if __name__ == "__main__":
    evalueaza_modelul()