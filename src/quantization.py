import numpy as np
from dct import inverse_dct
import matplotlib.pyplot as plt

def quantization(image):

    # Define the quantization table
    quantization_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Example 8x8 DCT block (Discrete Cosine Transform result)
    dct_block = np.array([
        [230, -23, -7, 12, -3, 0, 0, 0],
        [-34, 12, 8, -5, -2, 0, 0, 0],
        [-15, -9, 7, -3, 1, 0, 0, 0],
        [-3, -5, 3, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # Apply quantization
    quantized_block = np.round(dct_block / quantization_table)

    return quantized_block

def inverse_quantization(image):

    # Define the quantization table
    quantization_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    #Inverse quantization
    quantized_block = quantization(image)
    reconstructed_block = quantized_block * quantization_table

    #Inverse DCT
    inverse_dct(reconstructed_block)

    #Clipping
    reconstructed_block = np.clip(reconstructed_block, 0, 255)
    
    #Convert to 8 bit integers to process an image
    reconstructed_block = reconstructed_block.astype(np.uint8)

    # Display the reconstructed block as an image
    plt.imshow(reconstructed_block, cmap='gray')
    plt.title("Reconstructed 8x8 Block grayimage")
    plt.colorbar()
    plt.show()

    return reconstructed_block


