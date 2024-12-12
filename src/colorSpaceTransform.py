import matplotlib.pyplot as plt
import numpy as np;
from PIL import Image;

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

def ycbcr_2_rgb(image):
    return image
    
filename = "../inputImages/pixel.png"
img = Image.open(filename)
    
width, height = img.size
print('width :', width)
print('height:', height)

rgb_img = img.convert('RGB')

#Extracting the rgb values from each pixel then converting to YCbCr

ycbcr_pixels = []
count = 0

for y in range(height):
    for x in range(width):
        pixel = rgb_img.getpixel((x, y))
        r, g, b = pixel

        #for testing purposes
        #print(f'| R: {r:3} | G: {g:3} | B: {b:3} |')#

        Y = 0.299 * r + 0.587 * g + 0.114 * b
        Cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
        Cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
        ycbcr_pixels.append((Y, Cb, Cr))
        count+=1

        print(f'| Y: {Y:6.2f} | Cb: {Cb:6.2f} | Cr: {Cr:6.2f} |')
 
print(count)

def main():
    file_names = get_images()

    with Image.open(file_names[4]) as im:
        ycbcr = im.convert('YCbCr')
        # print(im.getchannel('Y'))
        y, cb, cr = ycbcr.split()

        cb_down = np.array(cb)[::2, ::2]
        cr_down = np.array(cr)[::2, ::2]

        cb_up = ds.upsample(cb_down, np.array(cb).shape)
        cr_up = ds.upsample(cr_down, np.array(cr).shape)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(ycbcr, cmap='gray')
    plt.title("YCbCr Image")
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.imshow(y, cmap='gray')
    plt.title("Y Channel")
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.imshow(cb_up, cmap='gray')
    plt.title("Cb Channel (after processing)")
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.imshow(cr_up, cmap='gray')
    plt.title("Cr Channel (after processing)")
    plt.colorbar()

    plt.show()

    print('in main')

if __name__ == '__main__':
    main()

