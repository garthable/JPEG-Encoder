
import numpy as np
from glob import glob

import matplotlib.pyplot as plt

from dct import *
from quantization import *

def main():
    # Example 8x8 block
    block = np.array([
        [230 - 5 * i - 5 * j for j in range(8)] for i in range(8)
    ])

    # Apply DCT
    dct_block = dct(block)

    # Quantize DCT coefficients
    quantized = quantization(dct_block)

    # Inverse Quantization
    dequantized = inverse_quantization(quantized)

    # Apply Inverse DCT
    # reconstructed_block = inverse_dct(dequantized)

    # Clip and convert to 8-bit
    # reconstructed_block = np.clip(reconstructed_block, 0, 255).astype(np.uint8)


    # Display original and reconstructed blocks
    plt.subplot(1, 4, 1)
    plt.imshow(block, cmap='gray')
    plt.title("Original Block")
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.imshow(dct_block, cmap='gray')
    plt.title("DCT Block")
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.imshow(quantized, cmap='gray')
    plt.title("Quantized Block")
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.imshow(dequantized, cmap='gray')
    plt.title("Dequantized Block")
    plt.colorbar()

    print(block)
    print(dct_block)
    print(quantized)
    print(dequantized)

    # assert dct_block.all() == quantized.all()

    plt.show()

if __name__ == "__main__":
    main()