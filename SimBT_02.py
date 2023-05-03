
################################################################################################################################################
#
# Steps to Create Application
#
# Step1: import library
# Step2: Image Loading Load the input image into program
# Step3: Grayscale Conversion : Convert the input image into grayscale format
# Step4: Invert Colors : Invert the grayscale image
# Step5: Gaussian Blur : Apply Guassian blur to smooth the image
# Step6: Edge Detection: Used Canny edge detector to detect edge in an image
# Step7: Threshoding : Apply Thresholding operation to edge detected image
# Step8: Dilation: Dilate Binary Image to thicken the edges
# Srep9: Final Sketch: Obtain Final Sketch

# PROBLEM STATEMENT: Write a program to convert a photo into photo sketch

################################################################################################################################################
#import library

import cv2
import matplotlib.pyplot as plt
import numpy as np

################################################################################################################################################
#Image Loading Load the input image into program

img = cv2.imread("image.jpeg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  #converting img to BGR to RGB
plt.figure(figsize=(7,7))
plt.imshow(img)
plt.title("Original Image")
plt.show()
cv2.imwrite("GrayScale_Image.png",img)

################################################################################################################################################
#Grayscale Conversion : Convert the input image into grayscale format
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(7,7))
plt.imshow(img_gray,cmap ="gray")
plt.title("GrayScale Image")
plt.show()
cv2.imwrite("GrayScale_Image.png",img_gray)

################################################################################################################################################
#Invert Colors : Invert the grayscale image
img_invert = cv2.bitwise_not(img_gray)
plt.figure(figsize=(7,7))
plt.imshow(img_invert,cmap="gray")
plt.title("Inverted Image")
plt.axis("off")
plt.show()
cv2.imwrite("Inverted_Image.png",img_invert)

################################################################################################################################################
#Gaussian Blur : Apply Guassian blur to smooth the image
img_smoothing = cv2.GaussianBlur(img_invert,(23,23),sigmaX = 0, sigmaY = 0)
plt.figure(figsize=(7,7))
plt.imshow(img_smoothing,cmap="gray")
plt.title("Blur Image")
plt.axis("off")
plt.show()
cv2.imwrite("Blur_Image.png",img_smoothing)

################################################################################################################################################
#Edge Detection: Used Canny edge detector to detect edge in an image

img = cv2.imread('image.jpeg', cv2.IMREAD_GRAYSCALE)
img_blur = cv2.GaussianBlur(img, (23,23), 0)

    # Calculate the gradient magnitude and direction using Sobel operator
sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
grad_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # Apply non-maximum suppression to thin out the edges
grad_dir[grad_dir < 0] += 180
edge_img = np.zeros_like(img)
for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        if (0 <= grad_dir[i,j] < 22.5) or (157.5 <= grad_dir[i,j] <= 180):
            if (grad_mag[i,j] > grad_mag[i,j-1]) and (grad_mag[i,j] > grad_mag[i,j+1]):
                edge_img[i,j] = grad_mag[i,j]
        elif (22.5 <= grad_dir[i,j] < 67.5):
            if (grad_mag[i,j] > grad_mag[i-1,j-1]) and (grad_mag[i,j] > grad_mag[i+1,j+1]):
                edge_img[i,j] = grad_mag[i,j]
        elif (67.5 <= grad_dir[i,j] < 112.5):
            if (grad_mag[i,j] > grad_mag[i-1,j]) and (grad_mag[i,j] > grad_mag[i+1,j]):
                edge_img[i,j] = grad_mag[i,j]
        else:
            if (grad_mag[i,j] > grad_mag[i-1,j+1]) and (grad_mag[i,j] > grad_mag[i+1,j-1]):
                edge_img[i,j] = grad_mag[i,j]

################################################################################################################################################
#Threshoding : Apply Thresholding operation to edge detected image
high_thresh, low_thresh = np.percentile(edge_img[edge_img > 0], (90, 30))
strong_edges = (edge_img > high_thresh)
weak_edges = (edge_img >= low_thresh) & (edge_img <= high_thresh)
non_edges = (edge_img < low_thresh)

################################################################################################################################################
#Dilation: Dilate Binary Image to thicken the edges
strong_edges = np.uint8(strong_edges)
weak_edges = np.uint8(weak_edges)
connected_edges = cv2.connectedComponents(strong_edges)[1]
for i in range(1, connected_edges.max()+1):
    points = np.argwhere(connected_edges == i)
    if edge_img[points[0,0], points[0,1]] > high_thresh:
        connected_edges[(connected_edges == i)] = 255
    else:
        connected_edges[(connected_edges == i)] = 0
connected_edges = np.uint8(connected_edges)
plt.figure(figsize=(7,7))
plt.imshow(connected_edges,cmap="gray")
plt.title("Edge Detection and Thresholding")
plt.axis("off")
plt.show()
cv2.imwrite('Edge_Detection_and_Thresholding.png', connected_edges)


################################################################################################################################################
#Final Sketch: Obtain Final Sketch
final_img = cv2.divide(img_gray,255-img_smoothing,scale =255)
plt.figure(figsize=(7,7))
plt.imshow(final_img,cmap="gray")
plt.title("final Sketch Image")
plt.axis("off")
plt.show()
cv2.imwrite("final_img.png",final_img)

################################################################################################################################################

#Reccomend to select the Image name as "image.jpeg" or Paste te name of image in the program
