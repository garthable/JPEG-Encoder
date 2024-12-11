import matplotlib.pyplot as plt



def plot_image(img):
  # Display the images
  fig, ax = plt.subplots(figsize=(10, 10))
  ax.imshow(img)
  ax.axis('off')
  plt.show()

def plot_rgb(img):
  # Display RGB Channels of our image
  fig, axs = plt.subplots(1, 4, figsize=(15, 5))
  axs[0].imshow(img)
  axs[1].imshow(img[:,:,0], cmap='Reds')
  axs[2].imshow(img[:,:,1], cmap='Greens')
  axs[3].imshow(img[:,:,2], cmap='Blues')
  axs[0].axis('off')
  axs[1].axis('off')
  axs[2].axis('off')
  axs[3].axis('off')
  axs[0].set_title('RGB')
  axs[1].set_title('Red channel')
  axs[2].set_title('Green channel')
  axs[3].set_title('Blue channel')
  plt.show()

def plot_ycbcr(ycbcr_img):
  # Extract and display YCbCr channels
  Y_channel = ycbcr_img[:, :, 0]
  Cb_channel = ycbcr_img[:, :, 1]
  Cr_channel = ycbcr_img[:, :, 2]

  fig, axs = plt.subplots(1, 4, figsize=(15, 5))
  
  axs[0].imshow(ycbcr_img)
  axs[0].axis('off')
  axs[0].set_title('YCbCr')
  
  axs[1].imshow(Y_channel, cmap='gray')
  axs[1].axis('off')
  axs[1].set_title('Y (Luminance) Channel')

  axs[2].imshow(Cb_channel)
  axs[2].axis('off')
  axs[2].set_title('Cb (Chrominance) Channel')

  axs[3].imshow(Cr_channel, cmap='coolwarm')
  axs[3].axis('off')
  axs[3].set_title('Cr (Chrominance) Channel')

  plt.tight_layout()
  plt.show()

def plot_subsampled_channels(y, cb, cr):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(y, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Y Channel')
    axs[1].imshow(cb, cmap='coolwarm')
    axs[1].axis('off')
    axs[1].set_title('Subsampled Cb Channel')
    axs[2].imshow(cr, cmap='coolwarm')
    axs[2].axis('off')
    axs[2].set_title('Subsampled Cr Channel')
    plt.tight_layout()
    plt.show()


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


def print_image_attributes(im):
    # print('Filename:', im.filename)
    print('Format:', im.format)
    print('Mode:', im.mode)
    print('Width:', im.width)
    print('Height', im.height)
    print('Palette:', im.palette)
    print('Info:', im.info)
    print('Transparency channel:', im.has_transparency_data)




def rgb_to_ycbcr(image):
    # Define the transformation matrix and offset
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    offset = np.array([0, 128, 128])
    
    # Ensure the image is a NumPy array
    rgb_array = np.asarray(image, dtype=np.float32)
    
    # Apply the transformation
    ycbcr_array = rgb_array @ transform_matrix.T + offset
    
    # Clip the values to ensure they are within valid ranges
    ycbcr_array = np.clip(ycbcr_array, 0, 255).astype(np.uint8)
    
    return ycbcr_array

def get_images(file_path='inputImages'):
    file_names = glob(f'{file_path}/*') # Add the content of 'inputImages' to list of file names

    print(f'\nSuccessfully loaded', len(file_names), 'images:\n')

    for infile in file_names:
        with Image.open(infile) as im:
            print(infile, im.format, f"{im.size} x {im.mode}")
    
    print('')
    
    return file_names
