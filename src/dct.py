import numpy as np
import matplotlib.pyplot as plt

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

# Test example
def main():
    
    test_image = np.zeros((8, 8), dtype=int)

    for i in range(8):
        for j in range(8):
            test_image[i, j] = (i + j) * 16

    transformed_image = run_dct_on_image(test_image.copy())
    reconstructed_image = run_inverse_dct_on_image(transformed_image.copy())

    print("Original Image:")
    print(test_image)
    print("\nTransformed Image (DCT Coefficients):")
    print(transformed_image)
    print("\nReconstructed Image (After IDCT):")
    print(reconstructed_image)

    plt.subplot(1, 3, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title("Original Block")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(transformed_image, cmap='gray')
    plt.title("DCT Block")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("Inverse-DCT Block")
    plt.colorbar()

    plt.show()
    
if __name__ == "__main__":
    main()