import tensorflow as tf
from tensorflow.contrib.losses import metric_learning
from tensorflow.python.framework import ops
import numpy as np
import os

def get_train_test_dataset(path, batchsize, mini, split_ratio):
    """
    Creates a dataset of image paths and labels
    CPU-driven. GPU should only be used for training the model.
    @Arguments:
        path: path to directory
        batchsize: size of batch
        mini: minimum number of images per class. less than will be filtered out
        split_ratio: train-to-test size
    """
    with tf.device('/cpu:0'):
        base_path = os.path.expanduser(path)
        classes = [path for path in os.listdir(path) \
                        if os.path.isdir(os.path.join(base_path, path))]
        classes.sort()
        imagepaths = []
        labels = []

        # Build the images and labels lists. Skip any class with less than
        # `mini` number of faces
        for i in range(len(classes)):
            class_name = classes[i]
            faces = os.path.join(base_path, class_name)
            n_faces = len(os.listdir(faces))
            if n_faces >= mini:
                image_paths = _get_image_paths(faces)
                imagepaths.append(image_paths)
                labels.append(classes[i])

        imgpath_sz, classes_sz = len(imagepaths), len(classes)
        imp, cla = imagepaths[:int(split_ratio*imgpath_sz)], classes[:int(split_ratio*classes_sz)]
        test_imp, test_cla = imagepaths[int(split_ratio*imgpath_sz):], classes[int(split_ratio*classes_sz):]
        train_dataset = tf.data.Dataset.from_tensor_slices((imp, cla))
        test_dataset = tf.data.Dataset.from_tensor_slice((test_imp, test_cla))
        train_dataset.shuffle(len(imp))
        test_dataset.shuffle(len(test_imp))
        train_dataset.map(_resize_images, num_parallel_calls=4)
        test_dataset.map(_resize_images, num_parallel_calls=4)
        train_dataset.batch(batchsize)
        test_dataset.batch(batchsize)
        train_dataset = train_dataset.prefetch(1)
        test_dataset = train_dataset.prefetch(1)
        return train_dataset, test_dataset

def _resize_images(path, label):
    """
    To be mapped to dataset.
    Resizes image specified by path.
    """
    notimg = tf.read_file(path)
    img = tf.image.decode_jpeg(notimg, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize_images(img, [64,64])
    return img, label

def _get_image_paths(facedir):
    """
        @Args:
            facedir: directory of faces
    """
    image_paths = []
    if os.path.isdir(facedir):
        imgs = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in imgs]
    return image_paths


def filter_dataset(dataset, min_num_img_per_class):
    """
    Get rid of classes with a size < min
    """

