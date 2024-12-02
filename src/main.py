from imageProcessor import *
from colorSpaceTransform import *
from dct import *
from downSample import *
from quantization import *
from util import *
import os

inputImagesPath = 'inputImages'

for path in os.listdir(inputImagesPath):
    image = read_image(inputImagesPath + "/" + path)
    if image is type(None):
        continue

    # Encode

    image = rgb_2_ycbcr(image)
    image = down_sample(image)
    image = run_dct_on_image(image)
    image = quantization(image)
    write_image(image, path, output_type.ENCODED)

    # Decode

    image = inverse_quantization(image)
    image = run_inverse_dct_on_image(image)
    image = up_sample(image)
    image = ycbcr_2_rgb(image)
    write_image(image, path, output_type.DECODED)