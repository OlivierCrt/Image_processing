import cv2
import numpy as np
import matplotlib.pyplot as plt

pilone_base = cv2.imread("/home/python/Image_processing/TP3/Input_image/pylone.png", cv2.IMREAD_GRAYSCALE)

hauteur, largeur = pilone_base.shape

def filtre_median_2d(image, taille_fenetre=3):
    hauteur, largeur = image.shape
    demi_fenetre = taille_fenetre // 2
    image_filtree = np.copy(image)

    for i in range(demi_fenetre, hauteur - demi_fenetre):
        for j in range(demi_fenetre, largeur - demi_fenetre):
            voisinage = image[i - demi_fenetre : i + demi_fenetre + 1, 
                              j - demi_fenetre : j + demi_fenetre + 1]
            image_filtree[i, j] = np.median(voisinage)
    return image_filtree

def MSE(Q, I):
    Q = Q.astype(np.float64)
    I = I.astype(np.float64)
    mse = np.mean((Q - I) ** 2)
    return mse

N_values = np.arange(1000, 20001, 1000)  # De 1000 à 20000 par pas de 1000
mse_bruitee = []  
mse_filtree = [] 

for N in N_values:
    pilone = pilone_base.copy()
    coordonnees_blancs = np.random.randint(0, high=[hauteur, largeur], size=(N, 2))
    for (y, x) in coordonnees_blancs:
        pilone[y, x] = 255
    coordonnees_noirs = np.random.randint(0, high=[hauteur, largeur], size=(N, 2))
    for (y, x) in coordonnees_noirs:
        pilone[y, x] = 0

    mse_bruitee.append(MSE(pilone, pilone_base))

    image_filtree = filtre_median_2d(pilone, 3)

    mse_filtree.append(MSE(image_filtree, pilone_base))

plt.figure(figsize=(10, 6))
plt.plot(N_values, mse_bruitee, label="MSE (Image bruitée vs Image de base)", marker='o')
plt.plot(N_values, mse_filtree, label="MSE (Image filtrée vs Image de base)", marker='x')
plt.xlabel("Nombre de pixels bruités (N)")
plt.ylabel("Erreur quadratique moyenne (MSE)")
plt.title("Progression des erreurs quadratiques en fonction de N")
plt.legend()
plt.grid()
plt.show()