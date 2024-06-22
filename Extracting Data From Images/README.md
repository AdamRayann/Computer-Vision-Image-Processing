# Histograms, Matching, and Quantization 
## Problem Description
 In this task, we need to read grade histograms from images, identify digits, quantify histogram bins, and measure the number of students per bin. Using digit images, we will recognize the digits in the histogram images, calculate the relative bin heights in pixels, and retrieve the number of students each bin represents. The images have consistent pixel proportions, allowing us to hardcode search areas for all images.

## Implementation Steps
 #### a) Reading and Displaying the First Image
 Goal: Verify that we can read and display histogram images in grayscale.
 Steps:
Use a function to read images from the directory.
Display the first image in grayscale to ensure it is read correctly.
 #### b) Reading Digit Images
 Goal: Load digit images to recognize numbers in histograms.
 Steps:
Read digit images from their directory.
Ensure that digit images are loaded correctly by displaying them.
 #### c) Implementing Histogram-Based Pattern Matching (compare_hist)
 Goal: Implement histogram-based pattern matching to recognize digits in histograms.
 Steps:
Use the Earth Mover's Distance (EMD) to compare histograms.
Create sliding windows over the source image and compare each window's histogram with the target digit's histogram.
Return whether a region with EMD less than a specified threshold (260) is found.

![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/3c30c1db-076f-43b9-aa17-d889ff28255a)


 #### d) Recognizing Digits in the Histogram
 Goal: Identify the highest number along the vertical axis of the histogram.
 Steps:
Apply the compare_hist function to the histogram image with digit images from 9 to 0.
The first detected number is the recognized digit.
 #### e) Quantization and Thresholding
 Goal: Quantize the histogram image to simplify it for bar height measurement.
 Steps:
Quantize the image to reduce the number of gray levels.
Threshold the quantized image to convert it to black and white, simplifying the identification of histogram bars.
 #### f) Measuring Bar Heights
 Goal: Determine the height of each bar in the histogram.
 Steps:
Define a function to measure the height of each bar in pixels.
Iterate over the bars and calculate their heights based on the black and white thresholded image.
 #### g) Calculating Number of Students per Bin
 Goal: Calculate the number of students per bin using the recognized digit and bar heights.
 Steps:
Use the formula: #students-per-bin = round(max-student-num * bin-height / max-bin-height)
Calculate the number of students per bin using the recognized highest number and the measured bar heights.
## Results
 The implementation successfully transcribed the histograms into numerical values representing the number of students per bin. The outputs for the provided images were printed as expected.

 Example Output
For example, the output for one image might be: Histogram a.jpg gave 1, 2, 3, 4, 5, 6, 7, 8, 9.

## Conclusion
This exercise demonstrated the application of histogram matching, quantization, and digit recognition to transcribe grade histograms from images. The method involved several steps, including edge detection, histogram comparison, and bar height measurement, to achieve accurate transcription.

## Additional Notes
Parameter Tuning: Adjusting parameters like quantization levels and EMD threshold was crucial for accurate results.
Future Improvements: Further refine the digit recognition process and explore alternative quantization methods for better accuracy.
## Screenshots:
## input 1 :
![a](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/85c0deea-4d65-42b8-a08a-ab0acfe8e03f)

## output 1 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/89c41452-795a-4969-b7ff-42bb8545d129)

## input 2 :
![b](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/17900cf5-39a8-435a-ad17-91ddb41e3e70)

## output 2 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/781b80c5-558e-4370-9ca9-6ae685e8f8ce)

## input 3 :
![c](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/9cb29793-1a25-428f-954d-86151961fe21)

## output 3 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/5ba4a639-c682-4981-9346-f16de395894b)
## input 4 :
![d](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/0a029933-d74a-4a36-b89c-1302668af25a)

## output 4 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/5a953a90-9b74-4bdf-a15d-51a6797c76d9)

## input 5 :
![e](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/1159b16d-0c43-48ec-8209-3b333c89ccab)

## output 5 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/0e0684df-62e4-4204-83a0-fb1c13e2ec1f)

## input 6 :
![f](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/a2d98775-31e4-4065-bd3a-ef7f2dae8227)

## output 6 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/a1c6e466-cb7b-4d43-a1d3-aa91586b5a6e)

## input 7 :
![g](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/a545d49a-9913-4fe9-b167-8b97019900e4)

## output 7 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/b04a06ab-d98e-498c-8966-83bb560e620e)



