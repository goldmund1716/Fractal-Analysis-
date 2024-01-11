
#1) Importieren der Module

import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from scipy.optimize import curve_fit

#image_path = 'image.png'
#original_image = Image.open(image_path)
#grayscale_image = color.rgb2gray(np.array(original_image))

#2) Einlesen Bilder

#aus ursprünglicher .py
#ToDo
#Image in grayscale, dann in Farbe, wenn möglich, zweimal einlesen und ausgeben
##Bilder sollen sich wieder schließen

def show_images(*images, wait: int = 0) -> None:
    img_show = np.hstack([
        (cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image)
        for image in images
    ])
    cv2.imshow("image", img_show)
    cv2.waitKey(wait)


img = cv2.imread("1-Images\Sea Change.PNG", -1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img[:, :710]
img_gray = img_gray[:, :710]
print(img_gray.shape)

cv2.imshow('Image', img)
cv2.imshow('Image-Gray', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

y = img_gray.astype(np.float32)

#3) Berechnung-DFA
#scale_range = segment
#scales = segment_size
#fluct = flucFunc

#20-11-23) Ansatz1) nur mit i arbeiten, for-loop integrieren, gucken ob ein vernünftiger plot entsteht
          #Ansatz2) "zweidimensionale berechnung" , j auch integrieren

def dfa(data, segment):
    n = len(data)
    segment_size = 2 ** np.arange(2, segment)  #damit kann experimentiert werden
    flucFunc = np.zeros(len(segment_size))

    for i, scale in enumerate(segment_size):
        y = np.cumsum(data - np.mean(data))
        y = y / np.max(np.abs(y))

        reshaped_data = y[: n - (n % scale)].reshape((n // scale, scale))
        flucFunc[i] = np.mean(np.std(reshaped_data, axis=1))



    # Fit a line to the data and calculate the slope (alpha)
    coefficient = np.polyfit(np.log2(segment_size), np.log2(flucFunc), 1)
    alpha = coefficient[0]


    return alpha


#3.1) Integrieren des For-Loops aus dem ursprünglichen Paper

#3.2) Storing von Alpha Values

#scale range kann angepasst werden

alpha_values = []
for i in range(img.shape[0]):
    row_alpha = dfa(img[i, :], segment=10)      #damit kann experimentiert werden
    alpha_values.append(row_alpha)

average_alpha = np.mean(alpha_values)
print("Average DFA Alpha:", average_alpha)      #Wie lasse ich Wert automatisch in Excel speichern?


#4) Schritt einfügen, welcher DFA Output in ein Polynom umwandelt


#y   = np.random.randn(1000)

#y = img_gray.astype(np.float32)
#y = img.astype(np.float32)

#mean = y.sum() * (1 / (len(y) * len(y[0])))
#y = np.cumsum(data - np.mean(data))
#y = y / np.max(np.abs(y))

#y = img.flatten

data = img.flatten()

# Set the segment value for DFA

segment_value = 10

# Perform DFA and get the scaling exponent (alpha)
alpha = dfa(data, segment_value)

# Generate a power-law distribution for visualization
box_sizes = 2 ** np.arange(2, segment_value)
flucFunc = box_sizes ** alpha

# Plot the results in a power-law distribution
plt.figure(figsize=(8, 6))
plt.scatter(np.log2(box_sizes), np.log2(flucFunc), label='DFA')
plt.xlabel('log2(Box Size)')
plt.ylabel('log2(Fluctuation)')
plt.title('Detrended Fluctuation Analysis (DFA) - Power Law Distribution')
plt.legend()
plt.show()


#5) Plotting

plt.plot(alpha_values)
plt.title('DFA Alpha Values Across Rows')
plt.xlabel('Row Index')
plt.ylabel('DFA Alpha')
plt.show()
plt.close()

#ToDo)
#Mehr Plots
#Literaturwerte vergleichen
#Plot mit Polynom

#Plot 1) Row Index over DFA Alpha
#Plot 2) Power Law Distrubution
#Also, the power law distribution is incredibly technical, and
#this project only looks to make sure that the results mirror this
#shape, but further work would take a deeper dive into the
#concept.)

#Plot 3)
#One example would be to include results
#graphed on a logarithmic scale to visually verify that the
#results graph in a line, and run the algorithm on images that
#don’t exhibit fractal behavior or have different fractal values

#Ziel am Ende:
#Ein Dataset an Bildern einlesen und zu jedem Bild Plots erstellen