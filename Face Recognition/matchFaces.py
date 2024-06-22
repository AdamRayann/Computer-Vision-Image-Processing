# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import warnings

from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore")


def enhance_contrast_clahe(image, clip_limit=2.0, tile_grid_size=(5, 5)):
    """
    Enhance contrast using CLAHE.

    Parameters:
    - image: Input image in grayscale or BGR format.
    - clip_limit: Threshold for contrast limiting.
    - tile_grid_size: Size of grid for histogram equalization. Smaller tiles result in more localized contrast enhancement.

    Returns:
    - Enhanced image.
    """
    # Convert to grayscale if input is BGR
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE on the grayscale image
    enhanced_image = clahe.apply(gray)

    return image


def reduce_noise_median(image, kernel_size=2):
    """
    Reduce noise using Median Filtering.

    Parameters:
    - image: Input image in grayscale or BGR format.
    - kernel_size: Size of the kernel. It must be an odd number.

    Returns:
    - Noise-reduced image.
    """
    return cv2.medianBlur(image, kernel_size)



def enhance_image(image):
    # Apply bilateral filter to reduce noise while preserving edges
    filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply unsharp mask for sharpening
    sharpened_image = cv2.addWeighted(image, 2, filtered_image, -1, 0)

    return sharpened_image


def scale_down(image, resize_ratio):
    # Perform the Fourier transform of the image
    fft_image = fft2(image)
    fft_shifted = fftshift(fft_image)

    # Calculate the new dimensions based on the resize ratio
    h, w = image.shape
    new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)

    # Calculate the cropping coordinates
    start_row, start_col = (h - new_h) // 2, (w - new_w) // 2
    end_row, end_col = start_row + new_h, start_col + new_w

    # Crop the centered frequency domain to achieve downscaling
    cropped_fft_shifted = fft_shifted[start_row:end_row, start_col:end_col]

    # Inverse shift and inverse FFT to get the downscaled image
    downscaled_image = np.abs(ifft2(ifftshift(cropped_fft_shifted)))

    return downscaled_image


def scale_up(image, resize_ratio):
    # Perform the Fourier transform of the image
    fft_image = fft2(image)
    fft_shifted = fftshift(fft_image)

    # Original dimensions
    h, w = image.shape

    # New dimensions, scaled up
    new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)

    # Initialize a new, larger spectrum for the scaled-up image
    upscaled_fft_shifted = np.zeros((new_h, new_w), dtype=complex)

    # Calculate the insertion coordinates
    start_row, start_col = (new_h - h) // 2, (new_w - w) // 2
    end_row, end_col = start_row + h, start_col + w

    # Insert the original, shifted frequency content into the center of the new spectrum
    upscaled_fft_shifted[start_row:end_row, start_col:end_col] = fft_shifted

    # Adjust the Fourier coefficients to account for the larger image size
    upscaled_fft_shifted *= (resize_ratio ** 2)

    # Inverse shift and inverse FFT to get the scaled-up image
    upscaled_image = np.abs(ifft2(ifftshift(upscaled_fft_shifted)))

    return upscaled_image


def reduce_noise_gaussian(image, kernel_size=7,s=-1):
    """
    Reduce noise using Gaussian Blurring.

    Parameters:
    - image: Input image in grayscale or BGR format.
    - kernel_size: Size of the Gaussian kernel. It must be an odd number.

    Returns:
    - Noise-reduced image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), s)






def ncc_2d(image, pattern):
    # Ensure floating point operations for accuracy
    image = image.astype(np.float32)
    pattern = pattern.astype(np.float32)

    # Calculate the mean of the pattern
    mean_pattern = np.mean(pattern)

    # Use sliding window view to generate all possible windows from the image that are the same size as the pattern
    # Copy the image to avoid modifying the original image outside this function
    modified_image = image.copy()

    # Zero out everything outside the rows 10 to 700 range
    # Ensure you do not exceed the image bounds
    image_height = modified_image.shape[0]
    start_row = 30  # Start row

    # Zero out rows before start_row and after end_row
    modified_image[:start_row, :] = 0

    # Now apply sliding window view on the modified image
    windows = sliding_window_view(modified_image, pattern.shape)

    # Reshape windows to make it easier to compute NCC across all windows
    windows_reshaped = windows.reshape(-1, pattern.size)
    pattern_flat = pattern.flatten()

    # Compute means of the windows
    mean_windows = np.mean(windows_reshaped, axis=1)

    # Compute the numerator of NCC: the covariance between the pattern and each window
    numerator = np.sum((windows_reshaped - mean_windows[:, np.newaxis]) * (pattern_flat - mean_pattern), axis=1)

    # Compute the denominator of NCC: the product of standard deviations
    std_pattern = np.std(pattern_flat)
    std_windows = np.std(windows_reshaped, axis=1)
    denominator = std_pattern * std_windows * pattern.size

    # Calculate NCC, avoiding division by zero by setting those values to 0
    ncc_values = np.where(denominator != 0, numerator / denominator, 0)

    # Reshape the NCC values back to the correct image shape
    ncc_image = ncc_values.reshape(image.shape[0] - pattern.shape[0] + 1, image.shape[1] - pattern.shape[1] + 1)

    return ncc_image


def display(image, pattern):
    plt.subplot(2, 3, 1)
    plt.title('Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Pattern')
    plt.imshow(pattern, cmap='gray', aspect='equal')

    ncc = ncc_2d(image, pattern)

    plt.subplot(2, 3, 5)
    plt.title('Normalized Cross-Correlation Heatmap')
    plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar()
    cbar.set_label('NCC Values')

    plt.show()


def draw_matches(image, matches, pattern_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for point in matches:
        y, x = point
        top_left = (int(x - pattern_size[1] / 2), int(y - pattern_size[0] / 2))
        bottom_right = (int(x + pattern_size[1] / 2), int(y + pattern_size[0] / 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)

    plt.imshow(image, cmap='gray')
    plt.show()

    cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)




def find_matches(ncc, threshold):
    # Display the original image and pattern
    display(image, pattern_scaled)

    plt.imshow(ncc)
    plt.show()
    matches=[]
    for i in range (len(ncc)):
        for j in range (len(ncc[0])):
            if ncc[i][j] > threshold:
                matches.append(([i,j]))


    return np.array(matches)





'''

CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# Students #############
image_scale_parameter=2
pattern_scale_parameter=0.95
threshold=0.51

image_scaled = scale_up(image,image_scale_parameter)
pattern_scaled =  scale_down(pattern,pattern_scale_parameter)

pattern_scaled=reduce_noise_gaussian(pattern_scaled)

display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = find_matches(ncc,threshold)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

real_matches[:, 0] = np.floor(real_matches[:, 0] / image_scale_parameter).astype(np.int32)
real_matches[:, 1] = np.floor(real_matches[:, 1] / image_scale_parameter).astype(np.int32)
# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"



'''

############# Crew #############
CURR_IMAGE = "thecrew"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

image_scale_parameter = 3.5
pattern_scale_parameter = 0.6
threshold=0.42


image_scaled = scale_up(image,image_scale_parameter)
pattern_scaled =  scale_down(pattern,pattern_scale_parameter)

pattern_scaled=reduce_noise_gaussian(pattern_scaled,1,s=0.5)

display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = find_matches(ncc, threshold)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

real_matches[:, 0] = np.floor(real_matches[:, 0] / image_scale_parameter).astype(np.int32)
real_matches[:, 1] = np.floor(real_matches[:, 1] / image_scale_parameter).astype(np.int32)
# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"

