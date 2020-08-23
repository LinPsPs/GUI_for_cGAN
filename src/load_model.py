from glob import glob
import numpy as np
from matplotlib import pylab as plt
import cv2
import tensorflow as tf
print(tf.__version__)
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import time
import os
from keras.models import load_model
import warnings
from warnings import simplefilter

warnings.filterwarnings("ignore")
simplefilter(action='ignore', category=FutureWarning)

path = 'C:/Users/Oemx/Desktop/datasets_91717_212894_data_val_1075086.png'

img_A = []
img1 = cv2.imread(path)
img1 = img1[..., ::-1]
img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
img_A.append(img1)
img_A = np.array(img_A) / 127.5 - 1

generator = load_model('C:/Users/Oemx/Desktop/Final/run_02/Ver_02.h5')

fake_A = generator.predict(img_A)

# Rescale images 0 - 1
gen_imgs = 0.5 * fake_A + 0.5

fig = plt.figure(figsize=(5, 5))
plt.imshow(gen_imgs[0])
plt.axis('off')
plt.savefig("C:/Users/Oemx/Desktop/test.png")
plt.show()
