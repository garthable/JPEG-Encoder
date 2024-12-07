import numpy as np

# Define the updated functions
def rgb_2_ycbcr(image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with 3 channels.")

    # Transformation matrix for RGB to YCbCr
    rgb_to_ycbcr_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    offset = np.array([0, 128, 128])

    # Reshape and apply transformation
    ycbcr = np.dot(image, rgb_to_ycbcr_matrix.T) + offset
    return ycbcr


def ycbcr_2_rgb(image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a YCbCr image with 3 channels.")

    # Transformation matrix for YCbCr to RGB
    ycbcr_to_rgb_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])

    offset = np.array([0, 128, 128])

    # Reshape and apply transformation
    rgb = np.dot(image - offset, ycbcr_to_rgb_matrix.T)
    return np.clip(rgb, 0, 255)  # Ensure values remain in valid range

# # Test with a mock RGB image (3x3 pixels)
# mock_rgb_image = np.array([
#     [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
#     [[255, 255, 0], [0, 255, 255], [255, 0, 255]],
#     [[128, 128, 128], [64, 64, 64], [192, 192, 192]]
# ], dtype=np.float32)

# # Step 1: Convert RGB to YCbCr
# ycbcr_image = rgb_2_ycbcr(mock_rgb_image)

# # Step 2: Convert YCbCr back to RGB
# reconstructed_rgb_image = ycbcr_2_rgb(ycbcr_image)

# # Display original RGB image
# print("Original RGB Image:")
# print(mock_rgb_image)

# # Display YCbCr conversion
# print("\nConverted YCbCr Image:")
# print(ycbcr_image)

# # Display Reconstructed RGB image
# print("\nReconstructed RGB Image:")
# print(reconstructed_rgb_image)