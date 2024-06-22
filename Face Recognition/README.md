# Face Detection:
### Problem Description:
The task is to detect all faces within given images using template matching. Specifically, we aim to implement various functions to scale images and compute normalized cross-correlation (NCC) for face detection. The goal is to process two different images with the same face template and accurately identify face locations.

### Implementation Details:
a) scale_down(image, resize_ratio)
This function scales down an image using a Fourier transform for optimal interpolation. The steps involved are:

 Blurring the image: To prevent aliasing, we blur the image before scaling it down.
Fourier Transform: We apply a Fourier transform to the image.
Scaling: The image is resized by the given ratio, retaining only the relevant part of the spectrum.
Inverse Fourier Transform: We transform the scaled image back to the spatial domain.
Tips: Initially implement with a ratio of 0.5 to understand the scaling effect.

b) scale_up(image, resize_ratio)
 This function scales up an image using a Fourier transform. The steps involved are similar to scaling down but in reverse:

 Fourier Transform: We apply a Fourier transform to the image.
 Scaling: The image is resized by the given ratio, expanding the relevant part of the spectrum.
 Inverse Fourier Transform: We transform the scaled image back to the spatial domain.
 Assumption: The resize ratio is at least 1.

c) ncc_2d(image, pattern)
 This function computes the normalized cross-correlation (NCC) between an image and a pattern. The steps are:

Sliding Window: Use np.lib.stride_tricks.sliding_window_view to create windows of the pattern's size across the image.
 NCC Calculation: For each window, compute the NCC. If the denominator (variance) is zero, set the cross-correlation to zero as the patch is constant.
d) display(image, pattern)
 This function displays the original image, the pattern, and their NCC heatmap. It helps visualize how well the pattern matches different parts of the image. We can choose to scale the image, the pattern, or both for optimal matching.

Example: A threshold value can be used to filter good matches. For instance, an NCC value of 0.2 is weak, and we might set a higher threshold to obtain significant matches.

e) draw_matches(image, matches)
This function draws red rectangles over recognized faces in the original image. It adjusts the matched locations relative to any scaling applied. The steps are:

 Filter Matches: Use a threshold to select significant matches from the NCC heatmap.
 Adjust Locations: If the image was scaled, adjust the match coordinates back to the original image size.
 Draw Rectangles: Draw rectangles around detected faces on the original image.
### Results:
The functions were used to detect faces in two images: students.jpg and thecrew.jpg. The output images, with detected faces highlighted, are saved in the same folder as the q2 folder.

### Output Images:
students_detected.jpg
thecrew_detected.jpg
### Conclusion:
The implementation successfully detected faces in the provided images using template matching. Scaling the images appropriately and computing NCC were crucial for accurate face detection. The final output images demonstrate the effectiveness of the approach.

### Additional Notes:
The choice of threshold for filtering matches significantly affects the detection results. Experiment with different values to optimize.
Consider the computational cost of Fourier transforms and sliding window operations for large images.
### Screenshots:
##### image 1 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/03e36d62-6437-4f43-905b-8fd2480195a1)

 ![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/9a5e794b-c047-437f-beba-92fe6dbe8ca5)

##### image 2 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/0d353378-342a-4480-af04-3df590c1c02f)

 ![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/6d52559d-33a8-4606-b97a-e2bbed35691e)



