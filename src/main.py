import pandas as pd
import numpy as np

from glob import glob

import matplotlib.pyplot as plt
import cv2

import util as ut

def main():
    files = glob('inputImages/*')
    images = []

    for f in range(len(files)):
        images.append(plt.imread(files[f]))

    print(f'\nSuccessfully loaded', len(files), 'images.')

    print("\n============ IMAGES ============")
    for f in range(len(files)):
        print(f'[{f}]:', files[f])
    
    choice = int(input('\nEnter the number of the image to use: '))

    print(files[choice], 'selected. Plotting images...')

    ut.plot_image(images[choice])
    ut.plot_rgb(images[choice])
    ut.plot_ycbcr(images[choice])

if __name__ == "__main__":
    main()