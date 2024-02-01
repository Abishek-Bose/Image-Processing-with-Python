import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import imageio
import scipy.ndimage
import numpy as np

import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS

def transform(images):
  return np.array(images)/127.5 - 1.
def inverse_transform(images):
  return (images+1.)/2
def prepare_data(sess, dataset):
  filenames = os.listdir(dataset)
  data_dir = os.path.join(os.getcwd(), dataset)
  data = glob.glob(os.path.join(data_dir, "*.bmp"))
  data = data + glob.glob(os.path.join(data_dir, "*.jpg"))
  return data

def imread(path, is_grayscale=False):
  if is_grayscale:
    return imageio.imread(path, flatten=True).astype(np.float)
  else:
    return imageio.imread(path).astype(np.float)

    
def imsave(image, path):

  imsaved = (inverse_transform(image)).astype(np.float)
  return imageio.imsave(path, imsaved)

def get_image(image_path,is_grayscale=False):
  image = imread(image_path, is_grayscale)

  return transform(image)

def get_lable(image_path,is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return image/255.
def imsave_lable(image, path):
  return imageio.imsave(path, image*255)
