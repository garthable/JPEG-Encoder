import matplotlib.pyplot as plt

def plot_image(img):
  # Display the images
  fig, ax = plt.subplots(figsize=(10, 10))
  ax.imshow(img)
  ax.axis('off')
  plt.show()

def plot_rgb(img):
  # Display RGB Channels of our image
  fig, axs = plt.subplots(1, 3, figsize=(15, 5))
  axs[0].imshow(img[:,:,0], cmap='Reds')
  axs[1].imshow(img[:,:,1], cmap='Greens')
  axs[2].imshow(img[:,:,2], cmap='Blues')
  axs[0].axis('off')
  axs[1].axis('off')
  axs[2].axis('off')
  axs[0].set_title('Red channel')
  axs[1].set_title('Green channel')
  axs[2].set_title('Blue channel')
  plt.show()

def plot_ycbcr(ycbcr_img):
  # Extract and display YCbCr channels
  Y_channel = ycbcr_img[:, :, 0]
  Cb_channel = ycbcr_img[:, :, 1]
  Cr_channel = ycbcr_img[:, :, 2]

  fig, axs = plt.subplots(1, 3, figsize=(15, 5))
  axs[0].imshow(Y_channel, cmap='gray')
  axs[0].axis('off')
  axs[0].set_title('Y (Luminance) Channel')
  axs[1].imshow(Cb_channel, cmap='coolwarm')
  axs[1].axis('off')
  axs[1].set_title('Cb (Chrominance) Channel')
  axs[2].imshow(Cr_channel, cmap='coolwarm')
  axs[2].axis('off')
  axs[2].set_title('Cr (Chrominance) Channel')
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
