# Image Compression Application

This Python application leverages the k-means clustering algorithm to perform image compression. The application is equipped with features to calculate the Within Cluster Sum of Squares (WCSS), Average Silhouette Score, and Calinski Harabasz Score for different values of k. Users can also open and compress their own images.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Features

1. **Image Compression:** Utilizes k-means clustering to compress images.
2. **Calculate WCSS:** Visualizes the Within Cluster Sum of Squares (WCSS) for different values of k.
3. **Average Silhouette Score:** Displays the average silhouette score for various values of k.
4. **Calinski Harabasz Score:** Shows the Calinski Harabasz score for different values of k.
5. **Open Custom Images:** Allows users to open and compress their own images.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ramya1907/image-compression-app.git

2. **Usage:**

   1. Open the application and click "Open Image" to select an image for compression. (Uses default image if image not selected)
   2. Enter the value of k (number of clusters) in the provided entry box.
   3. Click "Display Compressed Image" to view the compressed image.
   4. Use the combobox to select different analysis options like calculating WCSS, Average Silhouette Score, or Calinski Harabasz Score.
   5. Reset the values or quit the application as needed

3. **Dependencies:**
   
   Python, Numpy, Matplotlib, Scikit-learn, Pillow (Python Imaging Library)



   
