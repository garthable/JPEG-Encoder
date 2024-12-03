import rawpy as rp
import imageio.v3 as iio
import numpy as np
from enum import Enum

class output_type(Enum):
    ENCODED=0,
    DECODED=1

def read_image(path: str) -> np.ndarray[int] | None:
    with rp.imread(path) as raw:
        return raw.postprocess()
    return None

def write_image(image: np.ndarray[np.ndarray[int]], name: str, type: output_type) -> None:
    destination = "outputImages(encoded)/"
    if type == output_type.DECODED:
        destination = "outputImages(decoded)/"
    iio.imwrite(destination + name + ".tiff", image)