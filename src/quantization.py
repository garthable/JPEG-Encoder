import numpy as np
from matplotlib import pyplot as plt
from dct import inverse_dct  

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

    # Perform inverse DCT
    reconstructed_block = inverse_dct(reconstructed_block)

    # Clip values to 0-255 range
    reconstructed_block = np.clip(reconstructed_block, 0, 255)

    # Convert to 8-bit unsigned integers
    reconstructed_block = reconstructed_block.astype(np.uint8)

    return reconstructed_block


# Example DCT block to test quantization and inverse quantization
if __name__ == "__main__":
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

    quantized = quantization(dct_block)
    reconstructed = inverse_quantization(quantized)
