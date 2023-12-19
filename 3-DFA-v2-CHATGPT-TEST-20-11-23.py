import numpy as np
import matplotlib.pyplot as plt
import cv2
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



def dfa(data, segment):
        n = len(data)
        segment_size = 2 ** np.arange(2, segment)  # damit kann experimentiert werden
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

# Load image data (replace 'your_image.jpg' with the actual image file)
# Make sure to install the required libraries: pip install pillow
from PIL import Image


# Create time series data
y = img.astype(np.float32)


# Apply Detrended Fluctuation Analysis (DFA)
box_sizes = np.logspace(1, np.log10(len(y) // 2), 20).astype(int)
log_box_sizes, log_fluctuations = dfa(y, box_sizes)

# Fit the Power Law Distribution
def power_law(x, alpha, beta):
    return alpha * (x ** beta)

params, covariance = curve_fit(power_law, np.exp(log_box_sizes), np.exp(log_fluctuations))
alpha, beta = params

# Plot the Results
plt.figure(figsize=(12, 8))

# Plot the original time series
plt.subplot(2, 2, 1)
plt.plot(time_series_data)
plt.title('Original Time Series')

# Plot the DFA results
plt.subplot(2, 2, 2)
plot_power_law_distribution(log_box_sizes, log_fluctuations)
plt.title('DFA - Power Law Distribution')

# Plot the fitted power-law distribution
plt.subplot(2, 2, 3)
plt.scatter(np.exp(log_box_sizes), np.exp(log_fluctuations), label='DFA')
x_fit = np.linspace(min(np.exp(log_box_sizes)), max(np.exp(log_box_sizes)), 100)
plt.plot(x_fit, power_law(x_fit, alpha, beta), 'r', label=f'Power Law Fit\nalpha={alpha:.2f}, beta={beta:.2f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Box Size')
plt.ylabel('Fluctuation')
plt.legend()
plt.title('Power Law Fit')

plt.tight_layout()
plt.show()
