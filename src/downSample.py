import numpy as np
import matplotlib.pyplot as plt

def downsample(image):
    """
    Downsamples the input image by taking every other pixel (4:2:0-like for one plane).
    """
    return image[::2, ::2]

def upsample(downsampled_image, original_shape):
    """
    Upsamples the downsampled image back to the original shape using nearest-neighbor interpolation.
    """
    h, w = original_shape
    upsampled_image = np.zeros((h, w), dtype=downsampled_image.dtype)
    for i in range(h):
        for j in range(w):
            upsampled_image[i, j] = downsampled_image[i // 2, j // 2]
    return upsampled_image

def main():

    test_image = np.zeros((8, 8), dtype=int)

    for i in range(8):
        for j in range(8):
            test_image[i, j] = (i + j) * 16

    downsampled = downsample(test_image)
    upsampled = upsample(downsampled, test_image.shape)

    # Display the results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title("Original Image")
    plt.colorbar(label="Intensity")

    plt.subplot(1, 3, 2)
    plt.imshow(downsampled, cmap='gray')
    plt.title("Downsampled Image")
    plt.colorbar(label="Intensity")

    plt.subplot(1, 3, 3)
    plt.imshow(upsampled, cmap='gray')
    plt.title("Upsampled Image")
    plt.colorbar(label="Intensity")

    plt.tight_layout()
    plt.show()

    print('\n', test_image, '\n\n', downsampled, '\n', upsampled)

if __name__ == "__main__":
    main()