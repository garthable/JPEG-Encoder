import numpy as np;
from PIL import Image;

def rgb_2_ycbcr(image):
    return image

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
