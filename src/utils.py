import matplotlib.pyplot as plt
import torchvision
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def visualize_dataloader(dataloader, classes, num_images=6):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images[:num_images]
    labels = labels[:num_images]

    # Create a grid from the batch of images
    img_grid = torchvision.utils.make_grid(images)

    # Show images
    imshow(img_grid)

    # Print labels
    print('Labels:', ' '.join(f'{classes[labels[j]]}' for j in range(num_images)))
