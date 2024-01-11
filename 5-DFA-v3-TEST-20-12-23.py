import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from scipy.optimize import curve_fit


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


def dfa(data, scale_range):
    n = len(data)
    scales = 2 ** np.arange(2, scale_range)
    fluct = np.zeros(len(scales))

    for i, scale in enumerate(scales):
        y = np.cumsum(data - np.mean(data))
        y = y / np.max(np.abs(y))

        reshaped_data = y[: n - (n % scale)].reshape((n // scale, scale))
        fluct[i] = np.mean(np.std(reshaped_data, axis=1))

    coeffs = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    alpha = coeffs[0]

    return alpha

def calculate_fractal_dimension(img, scale_range=10):
    alpha_values = []
    for i in range(img.shape[0]):
        row_alpha = dfa(img[i, :], scale_range=scale_range)
        alpha_values.append(row_alpha)

    average_alpha = np.mean(alpha_values)
    fractal_dimension = 2 - average_alpha

    #print("Average DFA Alpha:", average_alpha)

    return fractal_dimension


# Calculate fractal dimension
fractal_dimension_values = []
for i in range(img.shape[0]):
    row_fractal_dimension = calculate_fractal_dimension(img[i, :])
    fractal_dimension_values.append(row_fractal_dimension)

    #print("Fractal Dimension:", fractal_dimension_values )

# Plot in a log-log scale
scales = np.arange(img.shape[0])  # assuming each row is a scale
plt.figure(figsize=(10, 6))
plt.scatter(scales, fractal_dimension_values, color='blue', marker='o', label='Fractal Dimension')
plt.xscale('log')
plt.yscale('log')
plt.title('Fractal Dimension in Power-Law Distribution')
plt.xlabel('Scale (log)')
plt.ylabel('Fractal Dimension (log)')
plt.legend()
plt.show()
