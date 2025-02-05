import numpy as np
import cv2
import os
import time  
from LutSubSamp import LutSubSamp

############################################################
# Crampette Olivier TP1 05/02/2025
############################################################

############################################################
# QUESTION 1
############################################################
def quantize_image(image, LUT):
    """
    Quantizes an RGB image using the given LUT.

    Parameters:
        image (numpy.ndarray): The input RGB image.
        LUT (numpy.ndarray): The look-up table for quantization.

    Returns:
        numpy.ndarray: The quantized image.
    """
    h, w, _ = image.shape
    pixels = image.reshape((-1, 3))

    distances = np.linalg.norm(pixels[:, None, :] - LUT[None, :, :], axis=2)#euclidienne

    closest_indices = np.argmin(distances, axis=1)
    quantized_pixels = LUT[closest_indices]

    return quantized_pixels.reshape((h, w, 3)).astype(np.uint8)

def MSE(Q, I):
    Q = Q.astype(np.float64)
    I = I.astype(np.float64)
    
    mse = np.mean((Q - I) ** 2)

    return mse

input_folder = "/home/python/Image_processing/TP1/Images"
output_folder = "/home/python/Image_processing/TP1/Quantized_images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

N = 8  
LUT = LutSubSamp(N)
print(LUT)

start_time = time.time()



for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".bmp")):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"quantized_{filename}")
        
        image_start_time = time.time()

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        quantized_image = quantize_image(image, LUT)
        cv2.imwrite(output_path, cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))
        print(f"Image quantifiée sauvegardée : {output_path}")
        
        mse_value = MSE(quantized_image, image)
        print("Erreur quadratique :", mse_value)
        
        image_end_time = time.time()
        image_processing_time = image_end_time - image_start_time

        print(f"Temps de traitement pour {filename}: {image_processing_time:.4f} secondes")
        print("\n")

end_time = time.time()
total_processing_time = end_time - start_time
print(f"Temps total de traitement pour toutes les images: {total_processing_time:.4f} secondes")

print("Traitement terminé !")
