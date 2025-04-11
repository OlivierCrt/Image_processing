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
# def quantize_image(image, LUT):
#     """
#     Quantizes an RGB image using the given LUT.

#     Parameters:
#         image (numpy.ndarray): The input RGB image.
#         LUT (numpy.ndarray): The look-up table for quantization.

#     Returns:
#         numpy.ndarray: The quantized image.
#     """
#     h, w, _ = image.shape
#     pixels = image.reshape((-1, 3))

#     distances = np.linalg.norm(pixels[:, None, :] - LUT[None, :, :], axis=2)

#     closest_indices = np.argmin(distances, axis=1)
#     quantized_pixels = LUT[closest_indices]

#     return quantized_pixels.reshape((h, w, 3)).astype(np.uint8)

def MSE(Q, I):
    Q = Q.astype(np.float64)
    I = I.astype(np.float64)
    
    mse = np.mean((Q - I) ** 2)

    return mse

# input_folder = "/home/python/Image_processing/TP1/Images"
# output_folder = "/home/python/Image_processing/TP1/Quantized_images"

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# N = 32  
# LUT = LutSubSamp(N)
# print(LUT)

# start_time = time.time()



# for filename in os.listdir(input_folder):
#     if filename.endswith((".jpg", ".png", ".bmp")):
#         image_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, f"quantized_{filename}")
        
        

#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         image_start_time = time.time()
#         quantized_image = quantize_image(image, LUT)
#         image_end_time = time.time()
#         cv2.imwrite(output_path, cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))
#         print(f"Image quantifiée sauvegardée : {output_path}")
        
#         mse_value = MSE(quantized_image, image)
#         print("Erreur quadratique :", mse_value)
        
        
#         image_processing_time = image_end_time - image_start_time

#         print(f"Temps de traitement pour {filename}: {image_processing_time:.4f} secondes")
#         print("\n")

# end_time = time.time()
# total_processing_time = end_time - start_time
# print(f"Temps total de traitement pour toutes les images: {total_processing_time:.4f} secondes")

# print("Traitement terminé !")





###################################################################
                    #Exercice2#
###################################################################





def floyd_steinberg_dithering(image, LUT):
    """
    Applique le dithering de Floyd-Steinberg sur une image en couleur.

    Parameters:
        image (numpy.ndarray): L'image d'entrée en RGB.
        LUT (numpy.ndarray): La table de correspondance des couleurs.

    Returns:
        numpy.ndarray: L'image avec dithering appliqué.
    """
    h, w, _ = image.shape
    image = image.astype(np.float64)

    for y in range(h):
        for x in range(w):
            old_pixel = image[y, x].copy()
            closest_color = LUT[np.argmin(np.linalg.norm(LUT - old_pixel, axis=1))]
            image[y, x] = closest_color

            quant_error = old_pixel - closest_color

            if x < w - 1:
                image[y, x + 1] += quant_error * 7 / 16
            if y < h - 1:
                image[y + 1, x] += quant_error * 5 / 16
            if x > 0 and y < h - 1:
                image[y + 1, x - 1] += quant_error * 3 / 16
            if x < w - 1 and y < h - 1:
                image[y + 1, x + 1] += quant_error * 1 / 16

    return np.clip(image, 0, 255).astype(np.uint8)


input_folder = "/home/python/Image_processing/TP1/Images"
output_folder = "/home/python/Image_processing/TP1/Dithering"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

N = 256  
LUT = LutSubSamp(N)
print(LUT)

start_time = time.time()

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".bmp")):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"quantized_{filename}")
        

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_start_time = time.time()

        quantized_image = floyd_steinberg_dithering(image, LUT)
        image_end_time = time.time()
        cv2.imwrite(output_path, cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))
        print(f"Image quantifiée avec dithering sauvegardée : {output_path}")
        
        mse_value = MSE(quantized_image, image)
        print("Erreur quadratique :", mse_value)
        
        
        image_processing_time = image_end_time - image_start_time

        print(f"Temps de traitement pour {filename}: {image_processing_time:.4f} secondes")
        print("\n")

end_time = time.time()
total_processing_time = end_time - start_time
print(f"Temps total de traitement pour toutes les images: {total_processing_time:.4f} secondes")

print("Traitement terminé !")
