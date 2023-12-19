from PIL import Image
import numpy as np

# Load the image
image_path = "path/to/your/image.jpg"
img = Image.open(image_path)

# Convert the image to a numpy array
img_array = np.array(img)


gray_img = img.convert('L')
gray_array = np.array(gray_img)

threshold = 128
binary_img = gray_img.point(lambda p: p > threshold and 255)
binary_array = np.array(binary_img)


def box_count(image, box_size):
    count = 0
    for i in range(0, image.shape[0], box_size):
        for j in range(0, image.shape[1], box_size):
            if np.any(image[i:i+box_size, j:j+box_size]):
                count += 1
    return count

# Choose a range of box sizes
box_sizes = [2, 4, 8, 16, 32, 64]

# Calculate the box count for each box size
counts = [box_count(binary_array, size) for size in box_sizes]

# Plot the results
import matplotlib.pyplot as plt

plt.plot(np.log(box_sizes), np.log(counts), marker='o')
plt.xlabel('log(Box Size)')
plt.ylabel('log(Box Count)')
plt.title('Box Counting Method')
plt.show()


from scipy.stats import linregress

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(box_sizes), np.log(counts))

# Fractal dimension is the negative of the slope
fractal_dimension = -slope
print(f"Fractal Dimension: {fractal_dimension}")
