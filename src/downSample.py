import numpy as np

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