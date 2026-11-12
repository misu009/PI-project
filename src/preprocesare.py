import cv2
import numpy as np

def crop_roi(image_slice, threshold=15):
    """
    Detectează automat genunchiul și taie marginile negre.
    """
    # transformam imaginea în format 8-bit (0-255) necesar pentru OpenCV
    if image_slice.max() <= 1.0:
        img_uint8 = (image_slice * 255).astype(np.uint8)
    else:
        img_uint8 = image_slice.astype(np.uint8)
        
    # Creăm o mască binară (pixelii cu intensitate > 15 devin albi, restul negri)
    _, binary_mask = cv2.threshold(img_uint8, threshold, 255, cv2.THRESH_BINARY)
    
    # Căutăm contururile în masca binară
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image_slice # Fallback dacă felia este complet neagră
        
    # Izolăm cel mai mare contur (genunchiul propriu-zis)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Decupăm imaginea originală cu un padding de 5 pixeli
    pad = 5
    H, W = image_slice.shape
    y_start = max(0, y - pad)
    y_end = min(H, y + h + pad)
    x_start = max(0, x - pad)
    x_end = min(W, x + w + pad)
    
    return image_slice[y_start:y_end, x_start:x_end]

def aplica_pipeline_preprocesare(volum_rmn):
    """
    Aplică întregul pipeline pe un volum 3D RMN de forma (nr_felii, inaltime, latime).
    Returnează volumul preprocesat și normalizat.
    """
    felii_procesate = []
    
    # Iterăm prin fiecare slice a RMN-ului
    for i in range(volum_rmn.shape[0]):
        felie = volum_rmn[i]
        
        # 1. Tăiem fundalul inutil
        felie_taiata = crop_roi(felie)
        
        # 2. Redimensionăm la 224x224 (pentru ViT / SE-ResNet)
        felie_224 = cv2.resize(felie_taiata, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # Conversie pentru filtrele OpenCV
        if felie_224.max() <= 1.0:
            img_8bit = (felie_224 * 255).astype(np.uint8)
        else:
            img_8bit = felie_224.astype(np.uint8)
            
        # 3. Filtru Gaussian pentru atenuare zgomot
        felie_fara_zgomot = cv2.GaussianBlur(img_8bit, (3, 3), 0)
        
        # 4. Aplicare CLAHE pentru contrast adaptiv
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        felie_clahe = clahe.apply(felie_fara_zgomot)
        
        # Readucem felia în format Float32 (0.0 - 1.0) pentru rețea
        felie_float = felie_clahe.astype(np.float32) / 255.0
        felii_procesate.append(felie_float)
        
    volum_procesat = np.array(felii_procesate)
    
    # 5. Normalizare Z-Score pe tot volumul
    medie = np.mean(volum_procesat)
    deviatie = np.std(volum_procesat)
    
    if deviatie > 0:
        volum_final = (volum_procesat - medie) / deviatie
    else:
        volum_final = volum_procesat
        
    return volum_final