# Adam Rayan, 212689889
# Adan Hammoud, 213011398

# Please replace the above comments with your names and ID numbers in the same format.
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")


# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
    img_size = imgs_arr[0].shape
    res = []

    for img in imgs_arr:
        X = img.reshape(img_size[0] * img_size[1], 1)
        km = KMeans(n_clusters=n_colors)
        km.fit(X)
        img_compressed = km.cluster_centers_[km.labels_]
        img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

        res.append(img_compressed.reshape(img_size[0], img_size[1]))
    return np.array(res)


# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
    image_arrays = []
    lst = [file for file in os.listdir(folder) if file.endswith(formats)]
    for filename in lst:
        file_path = os.path.join(folder, filename)
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_arrays.append(gray_image)
    return np.array(image_arrays), lst


# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
    # Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
    x_pos = 70 + 40 * idx
    y_pos = 274
    while image[y_pos, x_pos] == 1:
        y_pos -= 1
    return 274 - y_pos


# Sections c, d
def compare_hist(src_image, target):
    # Calculate the histogram of the target image
    target_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()

    # Sliding window view of the source image
    window_shape = target.shape[:2]
    windows = np.lib.stride_tricks.sliding_window_view(src_image, window_shape)
    start_row, start_col = 114, 25
    end_row, end_col = 115, 50

    # Iterate through windows in the top region
    for hh in range(start_row, end_row):
        for ww in range(start_col, end_col):
            # Calculate the histogram of the current window
            window_hist = cv2.calcHist([windows[hh, ww]], [0], None, [256], [0, 256]).flatten()
            cdf1 = np.cumsum(window_hist)
            cdf2 = np.cumsum(target_hist)
            plt.subplot(122)

            # Calculate the Earth Mover's Distance (EMD) between histograms
            emd_distance = np.sum(np.abs(cdf1 - cdf2))  # Using L1 norm as a simple alternative to EMD
            # Check if the EMD distance is below the threshold (260)
            if emd_distance < 300:
                return True  # Region found
    return False  # No region found


def make_it_binary(src_image):
    threshold_value = 210  # Example threshold value for a 3-level quantized image
    # Apply thresholding to convert to black and white
    binary_image = np.zeros_like(src_image)  # Create an empty black image of the same size
    binary_image[:, :] = 255
    binary_image[src_image < 214] = 1  # Set pixels until the threshold to black
    # cv2.imshow(names[0], binary_image)  # ***
    return binary_image


# Sections a, b
images, names = read_dir('data')
numbers, _ = read_dir('numbers')
cv2.imshow(names[0], images[0])

# Section e
img = quantization(images, 4)
cv2.imshow("quantization", img[0])
a = make_it_binary(img[0])
cv2.imshow("black&white", a)

id = 0
while (id < 7):
    i = 9
    while (i >= 0):
        if (compare_hist(images[id], numbers[i])):
            # if(id==0):
            # print("The function compare_hist on the ", names[id], " with the number ", i, "detected successfully.")
            # print("The highest number along the vertical axis is ", i, "(a.jpg)")
            break
        else:
            # if(id==0):
            # print("The function compare_hist on the " ,names[id], " with the number ",i, "did not detected.")
            i -= 1
    max_student_num = i
    # quantizated_images = quantization(images)
    a = make_it_binary(img[id])
    my_list = [0] * 10
    i = 9
    max_bin_height = 0
    while (i >= 0):
        my_list[i] = get_bar_height(a, i)
        i -= 1

    max_bin_height = max(my_list)
    i = 9
    while (i >= 0):
        my_list[i] = round(max_student_num * my_list[i] / max_bin_height)
        # if (id == 0):
        # print("the height of bar ", i, " is", my_list[i])
        # print("Student's number of the bar ", i, " is", my_list[i])
        i -= 1

    # The following print line is what you should use when printing out the final result - the text version of each histogram, basically.
    print(f'Histogram {names[id]}  gave {",".join(map(str, my_list))}')
    id += 1
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
