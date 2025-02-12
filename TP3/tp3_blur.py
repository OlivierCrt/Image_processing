import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("/home/python/Image_processing/TP3/Input_image/temple.png", cv2.IMREAD_GRAYSCALE)





def flou(img, size=5):
    return cv2.GaussianBlur(img, (size, size), 0)


# Fonction pour appliquer l'algorithme du masque flou
def masque_flou(img, coef=1.5, flou_size=5):
    # I1 est l'image originale (img)
    I1 = img

    # I2 est le flou (image floutée)
    I2 = flou(I1, size=flou_size)

    # I3 est le masque de flou (différence entre I1 et I2)
    I3 = cv2.subtract(I1, I2)

    # I4 est l'amplification de la partie nette
    I4 = I3 * coef

    # Convertir I4 au même type que I1 (uint8)
    I4 = np.clip(I4, 0, 17).astype(np.uint8)

    # I5 est l'image finale, combinant I1 et I4
    I5 = cv2.add(I1, I4)

    # S'assurer que les valeurs sont dans la plage [0, 255]
    I5 = np.clip(I5, 0, 255)

    return I2, I3, I4, I5



# Définir le coefficient d'amplification et la taille du filtre de flou
coef = 1.5
flou_size = 5

# Appliquer l'algorithme du masque flou
I2, I3, I4, I5 = masque_flou(img, coef=coef, flou_size=flou_size)

# Afficher les résultats
plt.figure(figsize=(12, 12))

# Afficher l'image originale
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Image Originale (I1)')
plt.axis('off')

# Afficher l'amplification de la partie nette (I4)
plt.subplot(1, 3, 2)
plt.imshow(I4, cmap='gray')
plt.title(f'Masque flou amplifié (I4) - Coef {coef}')
plt.axis('off')

# Afficher l'image finale (I5)
plt.subplot(1, 3, 3)
plt.imshow(I5, cmap='gray')
plt.title('Image Finale (I5)')
plt.axis('off')

# Afficher la comparaison entre les images
plt.tight_layout()
plt.show()
