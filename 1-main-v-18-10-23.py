import pickle

import cv2
import numpy as np


def show_images(*images, wait: int = 0) -> None:
    img_show = np.hstack([
        (cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image)
        for image in images
    ])
    cv2.imshow("image", img_show)
    cv2.waitKey(wait)


img = cv2.imread(".\\image.jpg", -1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img[:, :710]
img_gray = img_gray[:, :710]
print(img_gray.shape)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

y = img_gray.astype(np.float32)

mean = y.sum() * (1 / (len(y) * len(y[0])))



with open("integrated_cache2.bin", "rb") as f:
    x = pickle.load(f)



integrated = x

segment_size = 5  # segment_size war vorher 20, dadurch gab es einen Error
assert segment_size >= 5
small = min(len(integrated), len(integrated[0]))
fluctuation_nums = []

# def segemetize(input_array, segment_size):
#     array = []
#
#     return array
#
# for segment in segemetize(integrated, segment_size):
#     unraveled = np.ravel(segment)
#
#     y_values = unraveled
#     x_values = list(range(segment_size ** 2))
#
#     coefficient = np.polynomial.polynomial.polyfit(x_values, y_values, 2)
#     f = np.poly1d(coefficient)


while segment_size <= (small / 4):
    flucFunc = 0
    for i in range(0, small, segment_size):
        for j in range(0, small, segment_size):
            segment = integrated[i:i + segment_size, j:j + segment_size]

            unraveled = np.ravel(segment)
            try:
                assert len(unraveled) == segment_size ** 2
            except:
                print(f"Happens at {i=} {j=} {len(unraveled)}")
                continue
            y_values = unraveled
            x_values = list(range(segment_size ** 2))

            coefficient = np.polynomial.polynomial.polyfit(x_values, y_values, 2)
            f = np.poly1d(coefficient)

            #13-11-23 An Loop arbeiten

            if i == 615 and j == 710:
                break
            print(i,j)
    print("Termination of the calculation of the while-loop")




i = 2 #11.10.23, vorher war keine Variable gesetzt, dementsprechend ging der Error dann weg
j = 2
segment = str("2")
#fluctuation = ["k","l"]
count = 3
flucFunc = 1


 # fluctuation value
for k in (range(i, i + (int(segment)))):
        for l in range(j, j + len(segment[0])):
            if k >= small or l >= small: break
            fluctuation[k][l] = (segment [k - i][l - j] - f(count))

q = 2
for k in range(i, i + len(segment)):
    for l in range(j, j + len(segment[0])):
        if k >= small or l >= small: break
        flucFunc += pow(fluctuation[k][l], q)
    flucFunc /= (len(fluctuation) * len(fluctuation[0]))
    flucFunc = pow(flucFunc, 1 / q)

#int vor (range..) & segment ausdruck gesetzt um tuple error zu beseitigen
#segment in "" gesetzt -> quasi in nen String umgewandelt

#plot

