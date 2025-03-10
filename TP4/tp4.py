import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image
image = cv2.imread('/home/python/Image_processing/TP4/input/medicament.png')
if image is None:
    # Créer une image de test si l'image n'existe pas
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (50, 50), 30, (255, 255, 255), -1)

# Convertir en niveaux de gris
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculer le gradient
kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

gradient_x = cv2.filter2D(gray_image, cv2.CV_32F, kernel_x)
gradient_y = cv2.filter2D(gray_image, cv2.CV_32F, kernel_y)
gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

# Normaliser le gradient pour l'affichage
gradient_display = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Algorithme de ligne de partage des eaux
def watershed_algorithm(gradient_image):
    rows, cols = gradient_image.shape
    labels = np.zeros((rows + 2, cols + 2), dtype=np.int32)
    processed_mask = np.ones((rows, cols), dtype=bool)
    flat_gradient = gradient_image.flatten()
    sorted_indices = np.argsort(flat_gradient)
    current_label = 2

    for idx in sorted_indices:
        i, j = idx // cols, idx % cols
        if processed_mask[i, j]:
            processed_mask[i, j] = False
            i_ext, j_ext = i + 1, j + 1
            neighbor_labels = set()
            for ni, nj in [(i_ext-1, j_ext), (i_ext+1, j_ext), 
                           (i_ext, j_ext-1), (i_ext, j_ext+1),
                           (i_ext-1, j_ext-1), (i_ext-1, j_ext+1),
                           (i_ext+1, j_ext-1), (i_ext+1, j_ext+1)]:
                if labels[ni, nj] > 1:
                    neighbor_labels.add(labels[ni, nj])
            if len(neighbor_labels) == 0:
                labels[i_ext, j_ext] = current_label
                current_label += 1
            elif len(neighbor_labels) == 1:
                labels[i_ext, j_ext] = list(neighbor_labels)[0]
            else:
                labels[i_ext, j_ext] = 0
    return labels[1:-1, 1:-1]

# Appliquer l'algorithme de ligne de partage des eaux
labels = watershed_algorithm(gradient_magnitude)

# Fonction pour calculer les intensités moyennes des régions
def compute_region_means(image, labels):
    unique_labels = np.unique(labels)
    region_means = {}
    for label in unique_labels:
        if label == 0:  # Ignorer les LPE
            continue
        region_pixels = image[labels == label]
        region_means[label] = np.mean(region_pixels)
    return region_means

# Fonction pour fusionner les régions



def merge_regions(labels, region_means, threshold=20):
    rows, cols = labels.shape
    new_labels = labels.copy()
    changed = True

    while changed:
        changed = False
        # Balayage horizontal
        for i in range(rows):
            for j in range(1, cols - 1):
                p1, p2, p3 = labels[i, j - 1], labels[i, j], labels[i, j + 1]
                if p1 != 0 and p3 != 0 and p1 != p3:
                    if abs(region_means[p1] - region_means[p3]) < threshold:
                        new_labels[new_labels == p3] = p1
                        changed = True
        # Balayage vertical
        for j in range(cols):
            for i in range(1, rows - 1):
                p1, p2, p3 = labels[i - 1, j], labels[i, j], labels[i + 1, j]
                if p1 != 0 and p3 != 0 and p1 != p3:
                    if abs(region_means[p1] - region_means[p3]) < threshold:
                        new_labels[new_labels == p3] = p1
                        changed = True
        # Mettre à jour les labels et les moyennes
        labels = new_labels.copy()
        region_means = compute_region_means(gray_image, labels)
    return labels

# Calculer les intensités moyennes des régions
region_means = compute_region_means(gray_image, labels)

# Fusionner les régions
merged_labels = merge_regions(labels, region_means, threshold=20)

# Afficher les résultats
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.title('Image originale')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(132)
plt.title('Ligne de partage des eaux')
plt.imshow(labels % 256, cmap='nipy_spectral')

plt.subplot(133)
plt.title('Après fusion des régions')
plt.imshow(merged_labels % 256, cmap='nipy_spectral')

plt.tight_layout()
plt.show()

# Nombre de régions restantes
unique_labels = np.unique(merged_labels)
num_regions = len(unique_labels) - 1  # Exclure les LPE (label 0)
print(f"Nombre de régions après fusion : {num_regions}")