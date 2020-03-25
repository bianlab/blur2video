import cv2
import torch
import numpy as np
from blur import genBlur
from matplotlib import pyplot as plt


def get_image(name, size, color=False, tensor=True, show=False):
    if color:
        image_BGR = cv2.imread(name)
        image_B = image_BGR[:, :, 0]
        image_G = image_BGR[:, :, 1]
        image_R = image_BGR[:, :, 2]
        image = np.array([image_B, image_G, image_R])
    else:
        image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2GRAY)
        # image = cv2.imread(name)[:, :, 2]
    image = cv2.resize(image, size)
    image = np.array(image).astype('float32') / 255.0
    if show:
        plt.figure()
        plt.imshow(image, 'gray')
        plt.show()
    if tensor:
        image = torch.from_numpy(image.astype('float32'))
        if color:
            image = image.unsqueeze(0)
        else:
            image = image.unsqueeze(0).unsqueeze(0)
    return image

