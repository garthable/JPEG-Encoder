import pandas as pd
import numpy as mp

from glob import glob

import matplotlib.pylab as plt
import cv2

def main():
    png_files = glob('inputImages/*.png')
    jpg_files = glob('inputImages/*.jpg')

    num_png_files = len(png_files)
    num_jpg_files = len(jpg_files)
    
    png_images = []
    jpg_images = []

    print(f'\nSuccessfully loaded', num_png_files,  'PNG image(s) and', num_jpg_files, 'JPG image(s).')
    
    while True:
        choice = input('\nSelect an image type (J/P): ')

        match choice:
            case 'j':
                print(f"\n=== JPG FILES ===")
                
                for f in range(num_jpg_files):
                    print(f'[{f}]:', jpg_files[f])
                
                choice = input('\nSelect an image: ')
                print(jpg_files[int(choice)], 'selected.')
                break


            case 'p':
                print(f"\n=== PNG FILES ===")

                for f in range(num_png_files):
                    print(f'[{f}]:', png_files[f])

                choice = input('\nSelect an image: ')
                print(jpg_files[int(choice)], 'selected.')
                
                print(f'Shape of the image:', jpg_images[choice])

                break
            case 'q':
                print('\nExiting...')
                break
            case _:
                print('Invalid input.')
    
if __name__ == "__main__":
    main()