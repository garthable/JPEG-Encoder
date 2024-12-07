import numpy as np

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
    return rgb_image