# Hough Triangles Report
## Exercise Description
This exercise involves implementing detectors for three specific types of triangles: equilateral, isosceles, and right triangles. The detection process is based on processing the standard Hough Transform (HT) map used for detecting lines. The implementation should handle various images with parameter tuning specific to each image but should run generically without manual interventions other than parameter changes.
### **Explanation The Solution** 
---

## 1) Detail the stages of the solution in short

1.  Image Preparation:

    * Input: Start with the input image.
    * Blurring: Apply a Gaussian blur to the image to reduce noise and improve the accuracy of edge detection.
    * Padding: Add padding to the image to ensure edges and lines near the borders are detected accurately.
    
2.  Edge Detection and Window Processing:

    * Edge Map: Calculate the edge map of the image using an edge detection algorithm such as Canny.
    * Windowing: Divide the edge map into windows of a specific size and process each window independently.

3.  Line Detection:

    * Hough Transform: Apply the Hough Transform to detect lines within each window. This yields a set of lines characterized by their (ρ, θ) parameters.
    
4.  Intersection Points:

    * Find Intersections: Identify points where these lines intersect and store these points in a list called intersections.

5.  Distance Matrix Calculation:

    * Initialize Matrix: Define a matrix distance_matrix with dimensions (len(intersections) x len(intersections)). Each element [i, j] in the matrix represents the Euclidean distance between the intersection points i and j.
      
6.  Validation of Connections:

    * Line and Edge Check: For each pair of intersection points:
     * Check if there is a line (from the lines detected in step 3) that connects these two points.
     * Verify if this line lies on the edge between the two points in the edge map.
     * If any condition is not met, update the corresponding value in the distance matrix to 0, indicating no direct connection.

7.  Triangle Detection:

    * Triangle Possibilities: Examine all possible combinations of three points from the intersection points.
    * Connection Check: For each set of three points, verify if they are connected (i.e., the distances between them are not 0 in the distance matrix). Store these connected sets as potential triangles.

8.  Triangle Classification:

    * Classify Triangles: Based on the lengths of the sides:
      * Equilateral Triangles: All three sides are of equal length.
      * Isosceles Triangles: Two sides are of equal length.
      * Right Triangles: The square of the longest side is equal to the sum of the squares of the other two sides (Pythagorean theorem).
    * Store Results: Organize the triangle points into three lists, each representing one of the triangle classes.
      
9.  Visualization:

    * Map Triangles: Map the triangle points onto the output image.
    * Color Edges: Color the edges of the triangles based on their classification.

_________________________________________________________________________________________________________________________________
## 2) Some problems or difficulties that I encountered and how I overcame them.

1.  Managing Excessive Edge Detection by Window-Based Processing:

    * The excessive edges in the image led the Hough Transform to detect too many lines, causing numerous false positives and complicating triangle identification. To solve this, we divided the image into smaller windows and processed each separately. This approach reduced the number of edges and lines in each section, making the detection process more manageable and accurate.

2.  Handling Noisy Data in Image Processing:
    
    * Noise in images caused numerous false positives in triangle detection using the Hough Transform. To address this, we applied Gaussian blurring to smooth out noise, added padding to prevent truncation of edges near borders, and implemented a validation step to discard lines not aligned with actual edges.
   
3.  Tuning Parameters for Best Result:

       * Finding the right parameters for edge detection and the Hough Transform was tricky, often leading to false positives or missed lines. To tackle this, I extensively tuned the parameters through iterative testing. I experimented with different edge detection thresholds and adjusted the Hough Transform settings. Using a validation set with known triangles, I refined the parameters to ensure accurate detection across various images.
   
4.  Handling Close Lines and Points:

    * Image processing resulted in multiple close lines and points, cluttering the data and complicating triangle identification. To address this, we removed duplicate lines, merged close lines, and clustered nearby intersection points. This reduced clutter and made the data more manageable for analysis.
_________________________________________________________________________________________________________________________________

## Screenshots :
### Image 1 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/d9859a1f-c9a2-4740-bac8-6f0033198510)

![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/6fefcb74-3029-4163-9e84-f1bc64d264c9)

### Image 2 :
![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/a759e77e-7bf7-4500-8836-bb0233bc0382)

![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/85a584ed-510b-4a4f-bc7c-de957edf5fa0)
 ### Image 3 :
 ![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/c038e407-1ba1-4754-8b80-7748404fcdf3)

![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/01a88927-046b-48a3-ad31-b56e22da6fb2)

 ### Image 4 :
 ![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/a0710384-8612-43cc-8383-cd0ff8316a5a)

 ![image](https://github.com/AdamRayann/Computer-Vision-Image-Processing/assets/129179113/dd5fea1b-fc20-4e8e-bfd8-0430203f127c)


