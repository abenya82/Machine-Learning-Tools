import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

def show_before_filter_after(image, afilter, padding, stride):
    # Perform convolution
    filtered_image = convolution_forward(image=image, kernel=afilter, padding=padding, stride=stride)

    # Create subplots in a single figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display before image
    axes[0].imshow(image)
    axes[0].set_title('Before')
    axes[0].axis('off')

    # Display filter heatmap
    axes[1].imshow(afilter, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Filter Heatmap')
    axes[1].axis('off')

    # Display after image
    axes[2].imshow(filtered_image)
    axes[2].set_title('After')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()




def create_multiple_filter_heatmaps(filter_values_list,filter_names_list, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    
    for i, ax in enumerate(axes.flat):
        if i < len(filter_values_list):
            im = ax.imshow(filter_values_list[i], cmap='viridis', interpolation='nearest')
            #cbar = plt.colorbar(im, ax=ax, shrink=.5)  # Adjust colorbar size here
            #cbar.ax.tick_params(labelsize=8)  # Adjust colorbar label size

            ax.set_title(filter_names_list[i])
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks
            ax.grid(True, color='grey', linestyle='-', linewidth=1)  # Turn on gridlines
        else:
            ax.axis('off')
    plt.tight_layout()
    return fig


def display_filter_heatmap(filter_values):
    plt.imshow(filter_values, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Filter Values Heatmap')
    
    # Set ticks at cell centers
    plt.xticks(np.arange(filter_values.shape[1]) - 0.5, np.arange(filter_values.shape[1]))
    plt.yticks(np.arange(filter_values.shape[0]) - 0.5, np.arange(filter_values.shape[0]))
    
    # Create gridlines between cells
    plt.grid(which='major', color='grey', linestyle='-', linewidth=1.5)
    plt.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    plt.show()


    
def display_image_filter_and_result(image, afilter, padding='same', stride=1):
        
    # Perform convolution
    filtered_image = convolution_forward(image=image, kernel=afilter, padding=padding, stride=stride)
    
    # Create subplots in a single figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the Gaussian filter
    axes[1].imshow(afilter, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Filter')
    axes[1].axis('off')

    # Plot the filtered image after convolution
    axes[2].imshow(filtered_image)
    axes[2].set_title('Filtered Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def padOne(image,pad):
    
    new_image = np.zeros([image.shape[0]+2*pad,image.shape[1]+2*pad,image.shape[2]])
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                new_image[i+pad][j+pad][k] = image[i][j][k];
    
    return new_image

def padMultiple(X,pad):
    image_list = []
    for image in X:
        image_list.append(padOne(image,pad))
    new_X = np.stack(image_list,axis=0)
    
    return new_X






def get_reference_window(image, i, j, k, window_size):
    reference_window = np.zeros((window_size, window_size))

    half_window = window_size // 2  # Calculate half window size
    for x in range(-half_window, half_window + 1):
        for y in range(-half_window, half_window + 1):
            reference_window[x + half_window][y + half_window] = image[i + x][j + y][k]
    
    return reference_window


def convolution_forward(image, kernel, padding='same', stride=1):
    if padding == 'same':
        pad_height_amount = kernel.shape[0] // 2
        pad_width_amount = kernel.shape[1] // 2
        image = np.pad(image, ((pad_height_amount, pad_height_amount), (pad_width_amount, pad_width_amount), (0, 0)), mode='constant')

    kernel_height, kernel_width = kernel.shape
    image_height, image_width, channels = image.shape

    # Calculate output dimensions
    new_image_height = (image_height - kernel_height) // stride + 1
    new_image_width = (image_width - kernel_width) // stride + 1

    new_image = np.zeros((new_image_height, new_image_width, channels))

    for c in range(channels):
        for i in range(0, new_image_height):
            for j in range(0, new_image_width):
                window = image[i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width, c]
                new_image[i, j, c] = np.sum(window * kernel)

    return new_image






def sobel_h_filter(size):
    if size % 2 == 0:
        raise ValueError("Size must be an odd number")
    half_size = size // 2
    filter_values = np.zeros((size, size))
    filter_values[half_size, :] = 2
    filter_values[:half_size, :] = -1
    filter_values[half_size + 1:, :] = 1
    
    return filter_values



def sobel_v_filter(size):
    if size % 2 == 0:
        raise ValueError("Size must be an odd number")
    half_size = size // 2
    filter_values = np.zeros((size, size))
    filter_values[:, half_size] = 2
    filter_values[:, :half_size] = -1
    filter_values[:, half_size + 1:] = 1

    return filter_values


def gauss_filter(size,sigma):
    g_filter = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)), (size, size))
    g_filter_final =  g_filter / np.sum(g_filter)
    return g_filter_final


def laplacian_filter(size,center_value=-4):
    if size % 2 == 0:
        raise ValueError("Size must be an odd number")
    
    filter_values = np.zeros((size, size))
    center = size // 2
    filter_values[center, center] = center_value  # Set the center value
    
    # Set the surrounding values
    filter_values[center - 1, center] = 1
    filter_values[center + 1, center] = 1
    filter_values[center, center - 1] = 1
    filter_values[center, center + 1] = 1
    
    return filter_values




def print_filter(filter_values):
    filter_df = pd.DataFrame(filter_values)
    pd.options.display.max_rows = None  # Display all rows
    pd.options.display.max_columns = None  # Display all columns
    with pd.option_context('display.colheader_justify','center'):
        print(filter_df.to_string(index=False, header=False))





def show_image_before_and_after(image,afilter,padding,stride):
    image1 = image
    image2 = convolution_forward(image=image,kernel=afilter,padding=padding,stride=stride)

    
    plt.figure(figsize=(10, 5)) 

    # First subplot (left)
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title('Before')
    plt.axis('off')

    # Second subplot (right)
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title('After')
    plt.axis('off') 

    plt.tight_layout() 
    plt.show()







def maxpool_image(image_tensor, kernel_size=2, stride=2):
    # shape (batch size, channels, height, width)
    if len(image_tensor.shape) != 4:
        raise ValueError("shape error")

    pooled_image = F.max_pool2d(image_tensor, kernel_size=kernel_size, stride=stride)

    return pooled_image


def avgpool_image(image_tensor, kernel_size=2, stride=2):
    # shape (batch size, channels, height, width) 
    if len(image_tensor.shape) != 4:
        raise ValueError("shape error")

    pooled_image = F.avg_pool2d(image_tensor, kernel_size=kernel_size, stride=stride)

    return pooled_image








def show_avgpool(image):

    image_np = np.array(image)
    image_np = np.expand_dims(image_np, axis=0)  # Add a batch dimension

    image_tensor = torch.tensor(image_np, dtype=torch.float)
    reshaped_tensor = image_tensor.permute(0, 3, 1, 2)  # Swap dimensions to (batch_size, channels, height, width)

    example_image_tensor = reshaped_tensor  

    # Apply max pooling to the example image tensor
    pooled_image = avgpool_image(example_image_tensor, kernel_size=2, stride=2)

    re_reshaped_tensor = pooled_image.permute(0, 2, 3, 1)

    plt.figure(figsize=(10, 5)) 

    # First subplot (left)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Before')
    plt.axis('off')

    # Second subplot (right)
    plt.subplot(1, 2, 2)
    plt.imshow(re_reshaped_tensor[0])
    plt.title('After')
    plt.axis('off') 

    plt.tight_layout() 
    plt.show()


def show_maxpool(image):

    image_np = np.array(image)
    image_np = np.expand_dims(image_np, axis=0)  # Add a batch dimension

    image_tensor = torch.tensor(image_np, dtype=torch.float)
    reshaped_tensor = image_tensor.permute(0, 3, 1, 2)  # Swap dimensions to (batch_size, channels, height, width)

    example_image_tensor = reshaped_tensor  

    # Apply max pooling to the example image tensor
    pooled_image = maxpool_image(example_image_tensor, kernel_size=2, stride=2)

    re_reshaped_tensor = pooled_image.permute(0, 2, 3, 1)

    plt.figure(figsize=(10, 5)) 

    # First subplot (left)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Before')
    plt.axis('off')

    # Second subplot (right)
    plt.subplot(1, 2, 2)
    plt.imshow(re_reshaped_tensor[0])
    plt.title('After')
    plt.axis('off') 

    plt.tight_layout() 
    plt.show()