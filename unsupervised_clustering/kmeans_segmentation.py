import numpy as np
import matplotlib.pyplot as plt
import cv2

def kmeans_segmentation(img, flattened_img, max_iterations=200, accuracy=0.9, k=5):
    # determine the criteria to fulfill before the kmeans segmentation algorithm stops
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, accuracy)
    _, labels, centers = cv2.kmeans(flattened_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    # reshape data into the original image shape
    segmented_image = segmented_data.reshape(img.shape)

    return segmented_image

if __name__ == "__main__":
    # Read in the image
    img = cv2.imread('images/pexels-katerina-holmes-5910804.jpg')

    # To perform k-means segmentation, imaged must be flattened to a 2d array with
    # 3 colour channels
    flattened_img = img.reshape((-1, 3))
    flattened_img = np.float32(flattened_img)
    # View the result of compressing the 3d image using numpy
    print("Before reshaping: ", img.shape)
    print("After reshaping: ", flattened_img.shape)

    segmented_image = kmeans_segmentation(img, flattened_img, k=10)

    cv2.imshow('Original image', img)
    cv2.imshow('Segmented image', segmented_image)
    cv2.waitKey(0)
