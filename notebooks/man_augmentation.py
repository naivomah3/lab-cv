import random
import numpy as np
import cv2

def apply_rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rotation, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def apply_random_noise(image):
    height, width, channel = image.shape
    mat = np.random.randn(height, width, channel) * random.randint(5, 30)
    return np.clip(image+mat, 0, 255).astype(np.uint8)

def apply_change_gamma(image, alpha=1.0, beta=0.0):
    return np.clip(alpha*image+beta, 0, 255).astype(np.uint8)

def apply_random_color(image, alpha=20):
    mat = [random.randint(-alpha, alpha), random.randint(-alpha, alpha),random.randint(-alpha, alpha)]
    return np.clip(image+mat, 0, 255).astype(np.uint8)

def apply_random_transformation(image):
    if np.random.randint(2):
        image = apply_change_gamma(image, random.uniform(0.8, 1.2), np.random.randint(100)-50)
    if np.random.randint(2):
        image = apply_random_noise(image)
    if np.random.randint(2):
        image = apply_random_color(image)
    return image