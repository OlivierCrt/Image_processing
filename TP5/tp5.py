import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the structuring element (cross)
croix = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

def erosion(image):
    
    hauteur, largeur = image.shape
    eroded = np.zeros_like(image)
    for y in range(1, hauteur - 1):
        for x in range(1, largeur - 1):
            region = image[y-1:y+2, x-1:x+2]
            eroded[y, x] = np.min(region[croix == 1])
    return eroded

def dilatation(image):
    
    hauteur, largeur = image.shape
    dilated = np.zeros_like(image)
    for y in range(1, hauteur - 1):
        for x in range(1, largeur - 1):
            region = image[y-1:y+2, x-1:x+2]
            dilated[y, x] = np.max(region[croix == 1])
    return dilated

def erosion_ultime(image):
    """
    Compute skeleton using the ultimate erosion method.
    """
    _, image_binaire = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    image_iter = image_binaire.copy()
    squelette = np.zeros_like(image)
    
    max_iterations = max(image.shape)  # Maximum number of iterations
    
    for i in range(max_iterations):
        # Erode the image
        eroded = erosion(image_iter)
        
        # Dilate the eroded image
        dilated_ero = dilatation(eroded)
        
        # Find the contours (difference between original and opened image)
        contours = cv2.subtract(image_iter, dilated_ero)
        
        # Add the contours to the skeleton
        squelette = cv2.bitwise_or(squelette, contours)
        
        # Update the image for next iteration
        image_iter = eroded.copy()
        
        # Stop if the image is completely eroded
        if cv2.countNonZero(image_iter) == 0:
            break
            
    return squelette

def compter_voisins(image):
    """
    Identify nodes and endpoints in a skeletonized image.
    Returns:
        noeuds: List of node positions as (y, x) tuples.
        extremites: List of endpoint positions as (y, x) tuples.
    """
    hauteur, largeur = image.shape
    
    noeuds = []
    extremites = []
    
    for y in range(1, hauteur - 1): 
        for x in range(1, largeur - 1):
            if image[y, x] == 255:  
                # 3x3 neighborhood (8-connectivity)
                voisins = [
                    image[y-1, x-1], image[y-1, x], image[y-1, x+1],
                    image[y, x-1],                  image[y, x+1],
                    image[y+1, x-1], image[y+1, x], image[y+1, x+1]
                ]

                # Convert to binary (0 and 1)
                voisins_binaires = [1 if v == 255 else 0 for v in voisins]
                
                # Count active neighbors
                degre = sum(voisins_binaires)
                
                if degre >= 3:
                    noeuds.append((y, x))  # Add node position
                elif degre == 1:
                    extremites.append((y, x))  # Add endpoint position
    
    return noeuds, extremites

def gradient_morphologique_interne(image):
    
    return cv2.subtract(image, erosion(image))

def gradient_morphologique_externe(image):
    
    return cv2.subtract(dilatation(image), image)

def gradient_symetrise(image):
    
    return cv2.bitwise_or(gradient_morphologique_externe(image), gradient_morphologique_interne(image))

def main():
    # Load images
    noirblanc_img = cv2.imread('TP5/input/noirblanc.png', cv2.IMREAD_GRAYSCALE)
    mont = cv2.imread('TP5/input/brouillard_mont.png', cv2.IMREAD_GRAYSCALE)
    
    # Binarize the grayscale image
    _, image_binaire = cv2.threshold(noirblanc_img, 127, 255, cv2.THRESH_BINARY)
    
    # # Part 1: Skeleton computation
    # squelette = erosion_ultime(noirblanc_img)
    # noeuds, extremites = compter_voisins(squelette)
    
    # Display skeleton results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(noirblanc_img, cmap='gray')
    plt.title('Image Originale')
    plt.axis('off')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(squelette, cmap='gray')
    # plt.title(f'Squelette (Noeuds: {len(noeuds)}, Extrémités: {len(extremites)})')
    # plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Part 3: Morphological gradients
    gradient_interne = gradient_morphologique_interne(mont)
    gradient_externe = gradient_morphologique_externe(mont)
    gradient_sym = gradient_symetrise(mont)
    
    # Display gradient results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(mont, cmap='gray')
    plt.title('Image Originale')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(gradient_interne, cmap='gray')
    plt.title('Gradient Morphologique Interne')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(gradient_externe, cmap='gray')
    plt.title('Gradient Morphologique Externe')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(gradient_sym, cmap='gray')
    plt.title('Gradient Morphologique Symétrisé')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Gradients Morphologiques", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Save results
    output_dir = 'TP5/output'
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, 'squelette.png'), squelette)
    cv2.imwrite(os.path.join(output_dir, 'gradient_interne.png'), gradient_interne)
    cv2.imwrite(os.path.join(output_dir, 'gradient_externe.png'), gradient_externe)
    cv2.imwrite(os.path.join(output_dir, 'gradient_symetrise.png'), gradient_sym)
    
    print(f"Les images ont été sauvegardées dans : {output_dir}")

if __name__ == "__main__":
    main()