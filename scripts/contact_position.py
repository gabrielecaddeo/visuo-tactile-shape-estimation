import numpy as np
import cv2
from scipy.cluster.vq import  kmeans, vq
import torch
import skimage
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import time
from  math import sqrt
from os.path import join
# Load the background image
back = cv2.imread('/path/to/the/background')

# Path to the folder containing the images
folder = '/path/to/the/folder'

# Define average pooling layers
average_conv = torch.nn.AvgPool2d(kernel_size=49, stride=1, padding=(24, 24))
average_conv2 = torch.nn.AvgPool2d(kernel_size=11, stride=1, padding=(5, 5))

# Loop through a range of images (assuming there are 90 images)
for i in range(0, 90):
    # Load the current image
    image = cv2.imread(join(folder,'Image_' + str(i) + '.png'))

    # Compute the difference between the current image and the background
    diff = image - back
    diff = diff + 127

    # Compute the L2 norm of the difference
    diff_norm = np.linalg.norm(diff, axis=2)

    # Normalize the difference values to the range [0, 1]
    min_val = np.min(diff_norm)
    max_val = np.max(diff_norm)
    scaled_data = (diff_norm - min_val) / (max_val - min_val)

    # Apply k-means clustering to segment the image
    centroids, mean = kmeans(scaled_data.flatten(), 2)
    centroids = np.sort(centroids)
    clusters, dist = vq(scaled_data.flatten(), centroids)
    clusters = clusters.reshape((320, 240))
    tens = torch.from_numpy(clusters).unsqueeze(0).type(torch.DoubleTensor)

    # Apply average pooling to smooth the segmentation output
    output = average_conv(tens)
    mask = output > 0.8
    output[mask] = 1
    output[~mask] = 0
    output2 = average_conv2(output)
    output = output.squeeze(0).numpy()
    mask = output2 > 0.95
    output2[mask] = 1
    output2[~mask] = 0
    output2 = output2.squeeze(0).numpy()

    # Find contours in the segmented image
    start = time.time()
    contours = find_contours(output, 0.9, 'high')
    contours = sorted(contours, key=lambda l: (len(l), l.any()))

    # Create a binary mask from the largest contour
    mask = skimage.draw.polygon2mask((320, 240), contours[-1])
    output3 = np.zeros_like(output2)
    output3[mask] = 1
    # print(time.time() - start)

    # Display the original image with overlaid contours
    fig, ax = plt.subplots()
    ax.imshow(output, cmap=plt.cm.gray)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    # Compute the center of the detected object
    x_center, y_center = np.argwhere(output3 == 1).sum(0) / np.count_nonzero(output3)
    print(x_center)
    print(y_center)

    # find the farest points to the center
    
    x_min = contours[-1][np.argmin(contours[-1][:,0]- x_center), :]
    y_min = contours[-1][np.argmin(contours[-1][:,1]- y_center), :]
    x_max = contours[-1][np.argmax(contours[-1][:,0]- x_center), :]
    y_max = contours[-1][np.argmax(contours[-1][:,1]- y_center), :]

    list_surface_points = [x_min, y_min, x_max, y_max]

    # Highlight the center of the object in the segmentation output
    output[int(x_center - 20):int(x_center + 20), int(y_center)] = 0.5
    output[int(x_center), int(y_center - 20):int(y_center + 20)] = 0.5

    # Find descrptive points of the contact patch
    i=0
    while i< len(list_surface_points):
        if (sqrt((x_center-list_surface_points[i][0])**2 + (y_center-list_surface_points[i][1])**2))<50:
            list_surface_points.pop(i)
        else:
            i +=1
    i =0
    j=1
    while j< len(list_surface_points):
        if (sqrt((list_surface_points[j][0]-list_surface_points[i][0])**2 + (list_surface_points[j][1]-list_surface_points[i][1])**2))<50:
            list_surface_points.pop(i)
        else:
            i +=1
            j = i+1

    for z in list_surface_points:

        output[int(z[0] - 20):int(z[0] + 20), int(z[1])] = 0.5
        output[int(z[0]), int(z[1] - 20):int(z[1] + 20)] = 0.5

 

    # Find the coordinates of the minimum value in the difference image
    x, y = np.unravel_index(np.argmin(diff_norm, axis=None), diff_norm.shape)

    # Display the concatenated images
    cv2.imshow('images', (np.concatenate((scaled_data, clusters.astype('float'), output, output2, output3), axis=1)))
    cv2.waitKey(4000)
