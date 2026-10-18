# PI-project
Computer Vision University Project

<table>
    <tr>
        <td>Nr.</td>
        <td>Autor(i) / An</td>
        <td>Titlul articolului / proiectului</td>
        <td>Aplicație / Domeniu</td>
        <td>Tehnologii utilizate</td>
        <td>Metodologie / Abordare</td>
        <td>Rezultate</td>
        <td>Limitări</td>
        <td>Comentarii suplimentare</td>
    </tr>
    <tr>
        <td>1</td>
        <td>Keiley Mead, Tom Cross, Greg Roger, Rohan Sabharwal, <br>Sahaj Singh, d Nicola Giannott, 2025</td>
        <td>MRI deep learning models for assisted diagnosis of <br>knee pathologies: a systematic review</td>
        <td>Diagnosticarea anomaliilor genunchiului <br>cu ajutorul imaginilor RMN</td>
        <td>Modele Deep Learning, rețele CNN (ResNet, VGG, DenseNet, DarkNet), <br>seturi de date RMN (2D și 3D, ex: MRNet)</td>
        <td>Analiza performanțelor algoritmilor DL pe diverse afecțiuni <br>(ACL, menisc), analizând impactul tehnicilor de data augmentation <br>și transfer learning.</td>
        <td>Modelele specifice pe o singură patologie sunt mai <br>precise decât cele de detecție generală.</td>
        <td>Lipsa standardizării datelor, absența unui diagnostic de referință clar</td>
        <td>Articolul sugerează să ne axăm pe o singură <br>afecțiune pentru a obține o acuratețe mai bună <br>și să definim extrem de clar metricele și ground-truth-ul.</td>
    </tr>
    <tr>
        <td>2</td>
        <td>Kowshik Deb Nath, A. F. M. Minhazur Rahman, <br>Md. Ali Hossain, 2023</td>
        <td>An Attention-Based Deep Learning Approach to <br>Knee Injury Classification from MRI Images</td>
        <td>Diagnosticarea anomaliilor genunchiului <br>cu ajutorul imaginilor RMN</td>
        <td>Rețele Neuronale Convoluționale, Mecanism de Atenție, Data Augmentation</td>
        <td>S-a extins CNN-ul de bază prin adăugarea unor module de atenție <br>după primele straturi de convoluție ca modelul să extragă selectiv <br>informațiile relevante și să elimine zgomotul. S-a folosit data augmentation <br>pentru combaterea overfitting-ului.</td>
        <td>Adăugarea modulului de atenție ajutat la obținerea <br>acurateții de 100% pentru clasificarea binară și de 91% <br>pentru multi-clasă. S-au calculat și metrice de Precizie, <br>Recall și F1-Score.</td>
        <td>Setul de date multi-clasă este dezechilibrat, iar modelul cu atenție este <br>mult mai complex computațional, având milioane de parametri antrenabili</td>
        <td>Am învățat că integrarea unui bloc Squeeze-and-Excitation <br>peste un CNN clasic crește capacitatea algoritmului de a <br>distinge leziunile subtile.</td>
    </tr>
    <tr>
        <td>3</td>
        <td>Priya Mittal, Bhisham Sharma, D.P. Yadav, Imed <br>Ben Dhaou, 2025</td>
        <td>Automated Knee Injury Detection in MRI Images Using <br>Vision Transformers</td>
        <td>Diagnosticarea anomaliilor genunchiului <br>cu ajutorul imaginilor RMN</td>
        <td>Vision Transformers(ViT), Transfer Learning, Binary Cross-Entropy Loss, <br>Data Augmentation.</td>
        <td>Înlocuirea totală a rețelelor CNN cu modele ViT care folosesc <br>mecanisme de self-attention. Imaginile RMN sunt împărțite în <br>fragmente, și se adăugă informații de poziție.</td>
        <td>Modelul ViT a depășit rețeaua CNN de bază<br>obținând: acuratețe de 96.5%, precizie de 95.8%, recall <br>de 94.6% și F1-score de 95.2%. </td>
        <td>Modelele Transformer necesită putere de calcul mare<br>față de modelele CNN clasice.</td>
        <td>Un algoritm de tip Vision Transformer va capta contextul global <br>mult mai bine decât un CNN standard.</td>
    </tr>
    <tr>
        <td>4</td>
        <td>Kyung Min Chung, 2023</td>
        <td>Deep Learning-Based Knee MRI Classification for <br>Common Peroneal Nerve Palsy with Foot Drop</td>
        <td>Diagnosticarea anomaliilor genunchiului <br>cu ajutorul imaginilor RMN</td>
        <td>EfficientNet-B5, ResNet152, VGG19, Transfer Learning, <br>Hărți termice Grad-CAM.</td>
        <td>Compararea a 3 arhitecturi CNN diferite. Decuparea spațiilor negre, <br>limitarea zonei analizate, redimensionare 224x224, normalizare Z-score, <br>fără data augmentation.</td>
        <td>EfficientNet-B5 a fost superior: AUC 0.946, acuratețe <br>88.1%, F1-Score 84.6%. Hărțile Grad-CAM au demonstrat <br>concentrarea precisă a modelului pe zona nervului lezat.</td>
        <td>Lipsa datelor despre momentul producerii leziunii (care afectează <br>aspectul RMN-ului); decuparea prea strictă a imaginilor a ignorat anumiți<br> mușchi utili; număr mic de pacienți.</td>
        <td>Demonstrează că un model echilibrat (EfficientNet) e mai bun <br>decât unul doar adânc.</td>
    </tr>
    <tr>
        <td>5</td>
        <td>Justin Bernard A. Carlos, 2021</td>
        <td>MRI Knee Image Enhancement using Image <br>Processing </td>
        <td>Diagnosticarea anomaliilor genunchiului <br>cu ajutorul imaginilor RMN</td>
        <td>Python, Algoritmul Canny Edge, Transformări morfologice (Eroziune), <br>Filtre (Gaussian).</td>
        <td>Conversia datelor din format brut k-space în imagini. Aplicarea unui <br>pipeline clasic de preprocesare (mască de ROI, detectarea <br>marginilor, eroziune) pentru atenuarea artefactelor.</td>
        <td>Reducerea vizuală a zgomotului de fond și a artefactelor<br>de mișcare din RMN, îmbunătățind claritatea trăsăturilor.</td>
        <td>Nu elimină complet artefactele. Set de date infim. Definirea manuală a <br>ROI-ului (hardcoded), nu automată.   </td>
        <td>Ajută la realizarea etapei de preprocesare (Image <br>Enhancement) din schema bloc a proiectului.</td>
    </tr>
</table>
