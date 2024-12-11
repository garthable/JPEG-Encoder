import numpy as np
from matplotlib import pyplot as plt
import dct as dc

def quantization(dct_block):
    """
    Apply JPEG quantization on an 8x8 DCT block.

    Parameters:
        dct_block (np.ndarray): 8x8 array representing DCT coefficients.

    Returns:
        np.ndarray: Quantized 8x8 DCT block.
    """
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

    # Apply quantization
    quantized_block = np.round(dct_block / quantization_table)

    return quantized_block


def inverse_quantization(quantized_block):
    """
    Perform inverse quantization and reconstruct an 8x8 DCT block.

    Parameters:
        quantized_block (np.ndarray): Quantized 8x8 DCT block.

    Returns:
        np.ndarray: Reconstructed 8x8 block after inverse quantization and inverse DCT.
    """
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

    # Perform inverse quantization
    reconstructed_block = quantized_block * quantization_table

    return reconstructed_block


# Example DCT block to test quantization and inverse quantization
def main():
    test_image = np.zeros((8, 8), dtype=int)
    dct_block = np.zeros((8, 8), dtype=int)

    for i in range(8):
        for j in range(8):
            test_image[i, j] = (i + j) * 16

    print("Original Image:")
    print(test_image)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title("Original Image")
    plt.colorbar(label="Intensity")

    dct_block = dc.run_dct_on_image(test_image)
    print("DCT Image:")
    print(dct_block)
    plt.subplot(1, 5, 2)
    plt.imshow(dct_block, cmap='gray')
    plt.title("DCT Image")
    plt.colorbar(label="Intensity")

    quantized = quantization(dct_block)
    print('Quantized Image')
    print(quantized)
    plt.subplot(1, 5, 3)
    plt.imshow(quantized, cmap='gray')
    plt.title("Quantized Image")
    plt.colorbar(label="Intensity")

    unquantized = inverse_quantization(quantized)
    print('Un-quantized Image')
    print(unquantized)
    plt.subplot(1, 5, 4)
    plt.imshow(unquantized, cmap='gray')
    plt.title("Un-quantized Image")
    plt.colorbar(label="Intensity")

    reconstructed = dc.inverse_dct(unquantized)
    reconstructed= np.clip(reconstructed, 0, 255)
    reconstructed = reconstructed.astype(np.uint8)
    print('Reconstructed Image')
    print(reconstructed)
    plt.subplot(1, 5, 5)
    plt.imshow(reconstructed, cmap='gray')
    plt.title("Reconstructed Image")
    plt.colorbar(label="Intensity")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    
    
