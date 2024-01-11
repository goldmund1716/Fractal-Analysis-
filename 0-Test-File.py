import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

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
    for i in range(img.shape[1]):  # Iterate over columns
        column_alpha = dfa(img[:, i], scale_range=scale_range)
        alpha_values.append(column_alpha)

    average_alpha = np.mean(alpha_values)
    fractal_dimension = 2 - average_alpha

    return fractal_dimension

# Load and preprocess the image
image_path = "1-Images\Sea Change.PNG"
original_image = Image.open(image_path)
grayscale_image = color.rgb2gray(np.array(original_image))

# Calculate fractal dimension
fractal_dimension_values = []
for i in range(grayscale_image.shape[1]):  # Iterate over columns
    column_fractal_dimension = calculate_fractal_dimension(np.expand_dims(grayscale_image[:, i], axis=1))
    fractal_dimension_values.append(column_fractal_dimension)

# Plot in a log-log scale
scales = np.arange(grayscale_image.shape[1])  # assuming each column is a scale
plt.figure(figsize=(10, 6))
plt.scatter(scales, fractal_dimension_values, color='blue', marker='o', label='Fractal Dimension')
plt.xscale('log')
plt.yscale('log')
plt.title('Fractal Dimension in Power-Law Distribution')
plt.xlabel('Scale (log)')
plt.ylabel('Fractal Dimension (log)')
plt.legend()
plt.show()
