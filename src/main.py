import numpy as np
from glob import glob
import cv2

import matplotlib.pyplot as plt

from dct import *
from util import *
from quantization import *

import downSample as ds

def main():
    file_names = get_images()

    with Image.open(file_names[4]) as im:
        rgb = im.convert('RGB')
        ycbcr = im.convert('YCbCr')


        rgb_split = rgb.split()
        ycbcr_split = ycbcr.split()

        cb_down = np.array(ycbcr_split[1])[::2, ::2]
        cr_down = np.array(ycbcr_split[2])[::2, ::2]

        cb_up = ds.upsample(cb_down, np.array(ycbcr_split[1]).shape)
        cr_up = ds.upsample(cr_down, np.array(ycbcr_split[2]).shape)

    # plt.figure(figsize=(16, 4))

    print('Original RGB image:')
    print(np.array(rgb_split))

    print('\nRGB Channels:')
    print('R channel')
    print(np.array(rgb_split[0]))

    print('G channel')
    print(np.array(rgb_split[1]))

    print('B channel')
    print(np.array(rgb_split[2]))

    print('\nYCbCr image:')
    print(np.array(ycbcr_split))

    print('\nYCbCr Channels:')
    print('\nY channel')
    print(np.array(ycbcr_split[0]))

    print('\nCb channel')
    print(np.array(ycbcr_split[1]))

    print('\nCr channel')
    print(np.array(ycbcr_split[2]))



    # plt.subplot(1, 4, 1)
    # plt.imshow(ycbcr, cmap='gray')
    # plt.title("YCbCr Image")
    # plt.colorbar()

    # plt.subplot(1, 4, 2)
    # plt.imshow(y, cmap='gray')
    # plt.title("Y Channel")
    # plt.colorbar()

    # plt.subplot(1, 4, 3)
    # plt.imshow(cb, cmap='gray')
    # plt.title("Cb Channel")
    # plt.colorbar()


    # plt.subplot(1, 4, 4)
    # plt.imshow(cr, cmap='gray')
    # plt.title("Cr Channel")
    # plt.colorbar()

    # plt.show()



    # plt.subplot(1, 4, 1)
    # plt.imshow(ycbcr, cmap='gray')
    # plt.title("RGB Image")
    # plt.colorbar()

    # plt.subplot(1, 4, 2)
    # plt.imshow(y, cmap='Reds')
    # plt.title("R Channel")
    # plt.colorbar()

    # plt.subplot(1, 4, 3)
    # plt.imshow(cb, cmap='Greens')
    # plt.title("G Channel")
    # plt.colorbar()

    # plt.subplot(1, 4, 4)
    # plt.imshow(cr, cmap='Blues')
    # plt.title("B Channel")
    # plt.colorbar()

    # plt.show()

    


if __name__ == "__main__":
    main()