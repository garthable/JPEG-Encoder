import numpy as np

from glob import glob

import matplotlib.pyplot as plt

# import cv2

import util as ut
import downSample as dt
import colorSpaceTransform as ct


def subsample(img):
    img1 = cv2.imread('g4g.png', 0) 
  
    # Obtain the size of the original image 
    [m, n] = img1.shape 
    print('Image Shape:', m, n) 
    
    # Show original image 
    print('Original Image:') 
    plt.imshow(img1, cmap="gray") 
    
    
    # Down sampling 
    
    # Assign a down sampling rate 
    # Here we are down sampling the 
    # image by 4 
    f = 4
    
    # Create a matrix of all zeros for 
    # downsampled values 
    img2 = np.zeros((m//f, n//f), dtype=np.int) 
    
    # Assign the down sampled values from the original 
    # image according to the down sampling frequency. 
    # For example, if the down sampling rate f=2, take 
    # pixel values from alternate rows and columns 
    # and assign them in the matrix created above 
    for i in range(0, m, f): 
        for j in range(0, n, f): 
            try: 
    
                img2[i//f][j//f] = img1[i][j] 
            except IndexError: 
                pass
    
    
    # Show down sampled image 
    print('Down Sampled Image:') 
    plt.imshow(img2, cmap="gray") 
    
    
    # Up sampling 
    
    # Create matrix of zeros to store the upsampled image 
    img3 = np.zeros((m, n), dtype=np.int) 
    # new size 
    for i in range(0, m-1, f): 
        for j in range(0, n-1, f): 
            img3[i, j] = img2[i//f][j//f] 
    
    # Nearest neighbour interpolation-Replication 
    # Replicating rows 
    
    for i in range(1, m-(f-1), f): 
        for j in range(0, n-(f-1)): 
            img3[i:i+(f-1), j] = img3[i-1, j] 
    
    # Replicating columns 
    for i in range(0, m-1): 
        for j in range(1, n-1, f): 
            img3[i, j:j+(f-1)] = img3[i, j-1] 
    
    # Plot the up sampled image 
    print('Up Sampled Image:') 
    plt.imshow(img3, cmap="gray") 

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

    print(files[choice], 'selected.')

    # UNCOMMENT TO PLOT 
    ##########################
    print('Plotting images...')
    ut.plot_image(images[choice])
    ut.plot_rgb(images[choice])
    ut.plot_ycbcr(images[choice])

    # ut.plot_subsampled_channels(dt.chroma_subsampling())

if __name__ == "__main__":
    main()