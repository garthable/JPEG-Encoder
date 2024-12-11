import numpy as np
import matplotlib.pyplot as plt

def chroma_subsampling(ycbcr_image, sampling_format="4:2:0"):
    """
    Apply chroma subsampling to a YCbCr image.
    Supported formats: 4:4:4, 4:2:2, 4:2:0
    """
    y, cb, cr = ycbcr_image[:, :, 0], ycbcr_image[:, :, 1], ycbcr_image[:, :, 2]
    if sampling_format == "4:4:4":
        # No subsampling
        return y, cb, cr
    elif sampling_format == "4:2:2":
        # Horizontal subsampling
        cb = cb[:, ::2]
        cr = cr[:, ::2]
    elif sampling_format == "4:2:0":
        # Horizontal and vertical subsampling
        cb = cb[::2, ::2]
        cr = cr[::2, ::2]
    else:
        raise ValueError("Unsupported sampling format")
    return y, cb, cr

def down_sample(image):
    return image

def up_sample(image):
    return image

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

    (downsampled, upsampled)

if __name__ == "__main__":
    main()