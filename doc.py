import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import cv2

from src.dct import *
from src.util import *
from src.quantization import *
from src.downSample import *

def downsample(image):
    return image[::2, ::2]

def upsample(downsampled_image, original_shape):
    h, w = original_shape
    upsampled_image = np.zeros((h, w), dtype=downsampled_image.dtype)
    for i in range(h):
        for j in range(w):
            upsampled_image[i, j] = downsampled_image[i // 2, j // 2]
    return upsampled_image

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

# Function to convert RGB to YCbCr
def rgb_to_ycbcr(rgb_image):
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    offset = np.array([0, 128, 128])
    ycbcr_image = np.dot(rgb_image, transform_matrix.T) + offset
    return ycbcr_image

# Function to convert YCbCr back to RGB
def ycbcr_to_rgb(ycbcr_image):
    inverse_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    offset = np.array([0, 128, 128])
    rgb_image = np.dot(ycbcr_image - offset, inverse_matrix.T)
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

def run_dct_on_image(image):
    subimages = segment_image(image)
    h, w = image.shape
    x, y = 0, 0
    for subimage in subimages:
        chunk = dct(subimage)
        for j in range(8):
            for k in range(8):
                image[y + j][x + k] = chunk[j][k]
        x += 8
        if x >= w:
            x = 0
            y += 8
    return image

def segment_image(image):
    h, w = image.shape
    assert h % 8 == 0, f"{h} rows is not evenly divisible by 8"
    assert w % 8 == 0, f"{w} cols is not evenly divisible by 8"
    return (image.reshape(h // 8, 8, -1, 8)
            .swapaxes(1, 2)
            .reshape(-1, 8, 8))

def C(z):
    return 1.0 / np.sqrt(2) if z == 0 else 1

def dctF(imageSeg, u, v):
    sum_val = 0.0
    for x in range(8):
        for y in range(8):
            a = np.cos((2 * x + 1) * u * np.pi / 16)
            b = np.cos((2 * y + 1) * v * np.pi / 16)
            sum_val += imageSeg[x][y] * a * b
    return 0.25 * C(u) * C(v) * sum_val

def dct(imageSeg):
    out = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            out[u][v] = dctF(imageSeg, u, v)
    return out

def run_inverse_dct_on_image(image):
    subimages = segment_image(image)
    h, w = image.shape
    x, y = 0, 0
    for subimage in subimages:
        chunk = inverse_dct(subimage)
        for j in range(8):
            for k in range(8):
                image[y + j][x + k] = chunk[j][k]
        x += 8
        if x >= w:
            x = 0
            y += 8
    return image

def inverse_dctF(imageSeg, x, y):
    sum_val = 0.0
    for u in range(8):
        for v in range(8):
            a = np.cos((2 * x + 1) * u * np.pi / 16)
            b = np.cos((2 * y + 1) * v * np.pi / 16)
            sum_val += C(u) * C(v) * imageSeg[u][v] * a * b
    return 0.25 * sum_val

def inverse_dct(imageSeg):
    out = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            out[x][y] = inverse_dctF(imageSeg, x, y)
    return out


def main():
    return

if __name__ == "__main__":
    main()