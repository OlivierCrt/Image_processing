import cv2
import numpy as np
import matplotlib.pyplot as plt

bateaunoir = cv2.imread("/home/python/Image_processing/TP2/Input_images/bateauNoir.bmp", cv2.IMREAD_GRAYSCALE)
pingu = cv2.imread("/home/python/Image_processing/TP2/Input_images/papaPingu.jpg", cv2.IMREAD_GRAYSCALE)
seanight = cv2.imread("/home/python/Image_processing/TP2/Input_images/seanight.bmp", cv2.IMREAD_GRAYSCALE)


def plot_histogram(image, title):
    histo = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histo, color='black')
    plt.title(title)
    plt.xlabel("Niveaux de gris")
    plt.ylabel("Nombre de pixels")
    plt.show()
    return histo



histobat=plot_histogram(bateaunoir, "Histogramme de bateaunoir")
histo_ping=plot_histogram(pingu, "Histogramme de pingu")
histo_sea=plot_histogram(seanight, "Histogramme de seanight")



def normalisation_et_histo(image):
    min_val, max_val = np.min(image), np.max(image)

    if max_val == min_val:
        return np.histogram(image, bins=256, range=(0, 255))[0]

    # recadrage dynamique formule
    image_norm = ((image - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(image_norm, cmap='gray')
    plt.title("Image Normalisée")
    plt.axis("off")

    histo = cv2.calcHist([image_norm], [0], None, [256], [0, 256])
    plt.subplot(1,2,2)
    plt.plot(histo, color='black')
    plt.title("Histogramme avec recadrage dynamique")
    plt.xlabel("Intensité")
    plt.ylabel("Nombre de pixels")
    plt.show()

    return histo

"""
normalisation_et_histo(bateaunoir)
normalisation_et_histo(pingu)
normalisation_et_histo(seanight)"""

def equalize_histogram_manual(image):
    """
    Effectue l'égalisation d'histogramme sur une image en niveaux de gris sans utiliser cv2.equalizeHist.
    
    Paramètre:
    - image: Image en niveaux de gris (2D array de type uint8)
    
    Retourne:
    - Image après égalisation d'histogramme.
    """
    
    
    histo = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    cdf = histo.cumsum()  
    hauteur, largeur= image.shape  
    N= hauteur*largeur
    
    cdf_normalized = cdf  * 255 / N

    cdf_normalized = cdf_normalized.astype(np.uint8)
    
    image_equalized = cdf_normalized[image]

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(image_equalized, cmap='gray')
    plt.title("Image Normalisée")
    plt.axis("off")

    histo = cv2.calcHist([image_equalized], [0], None, [256], [0, 256])
    plt.subplot(1,2,2)
    plt.plot(histo, color='black')
    plt.title("Histogramme avec equalized")
    plt.xlabel("Intensité")
    plt.ylabel("Nombre de pixels")
    plt.show()
    
    return image_equalized
"""
equalize_histogram_manual(pingu)
equalize_histogram_manual(bateaunoir)
equalize_histogram_manual(seanight)"""



def transfert(t):
    # Convertir t en une matrice avec 3 colonnes, une pour chaque composant RGB
    s = t.shape
    m = np.zeros((s[0], 3), dtype=float)
    
    m[:, 0] = t[:, 0]  # La première colonne pour le rouge
    m[:, 1] = t[:, 0]  # La deuxième colonne pour le vert
    m[:, 2] = t[:, 0]  # La troisième colonne pour le bleu
    
    # Normaliser les valeurs entre 0 et 1 pour la colormap
    m = m / 255.0
    
    # Appliquer la colormap
    plt.imshow(m, cmap='gray', aspect='auto')
    plt.axis('off')
    plt.show()

# Affichez l'image
plt.imshow(pingu, cmap='gray')
plt.axis('off')
plt.show()

# Appliquez le transfert avec la matrice de valeurs [0:255]
transfert(np.array([np.arange(0, 256)]).T)

