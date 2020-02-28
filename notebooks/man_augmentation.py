import random
import numpy as np
import cv2

def apply_rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rotation, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# def apply_random_noise(image):
#     height, width, channel = image.shape
#     mat = np.random.randn(height, width, channel) * random.randint(5, 30)
#     return np.clip(image+mat, 0, 255).astype(np.uint8)


# Automatic brightness and contrast optimization with optional histogram clipping
# Have good effect in noise removal
def apply_auto_br_ct(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # return (auto_result, alpha, beta)
    return auto_result


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def apply_change_gamma(image, alpha=.8, beta=0.0):
    return np.clip(alpha*image+beta, 0, 255).astype(np.uint8)

# def apply_random_color(image, alpha=10):
#     mat = [random.randint(-alpha, alpha), random.randint(-alpha, alpha),random.randint(-alpha, alpha)]
#     return np.clip(image+mat, 0, 255).astype(np.uint8)

def apply_random_transformation(image):
    if np.random.randint(2):
        image = apply_change_gamma(image, random.uniform(0.8, 1.2), np.random.randint(100)-50)
    if np.random.randint(2):
        image = apply_clahe(image)
    # if np.random.randint(2):
    #     image = apply_random_noise(image)
    # if np.random.randint(2):
    #     image = apply_random_color(image)
    return image