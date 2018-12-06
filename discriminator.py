import tensorflow as tf
from tensorflow.contrib.losses.metric_learning import triplet_semihard_loss
import numpy as np
import os
import cv2

class ImageSet():
    """ Stores class name and image paths for class"""
    def __init__(self, class_name, image_paths):
        self.name = class_name
        self.image_paths = image_paths

def get_dataset(path):
    dataset = [] # A dataset of ImageSets
    base_path = os.path.expanduser(path)
    classes = [path for path in os.listdir(path) \
                    if os.path.isdir(os.path.join(base_path, path))]
    classes.sort()
    for i in range(len(classes)):
        class_name = classes[i]
        faces = os.path.join(base_path, class_name)
        image_paths = _get_image_paths(faces)
        dataset.append(ImageSet(class_name, image_paths))

    return dataset

def _get_image_paths(facedir):
    """
        @Args:
            facedir: directory of faces
    """
    image_paths = []
    if os.path.isdir(facedir):
        imgs = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img im imgs]
    return image_paths

def get_batch(images, batch_size, curr_batch_index):
#

# Constructs a list of image paths
def get_image_paths(image_dir):
#

def get_distance_matrix(batch):
    batch_squared = tf.matmul(batch, tf.transpose(batch))
    squares = tf.diag_part(batch_squared)
    a2 = tf.expand_dim(squares, 0)
    b2 = tf.expand_dim(squares, 1)
    distances = a2 - 2 * batch_squared - b2

    return distances

"""
Triplet Loss function as described by the FaceNet Paper
    In contrast to these approaches, FaceNet directly trains
    its output to be a compact 128-D embedding using a tripletbased
    loss function based on LMNN [19]. Our triplets consist
    of two matching face thumbnails and a non-matching
    face thumbnail and the loss aims to separate the positive pair
    from the negative by a distance margin.

    Ensure that an image x_ia (anchor) of
    a specific person is closer to all other images x_ip
    (positive) of the same person than it is to any image x_ni
    (negative) of any other person.
"""
def triplet_loss(anchor, positive, negative, alpha):
    """
        @Arguments:
            anchor: base face
            positive: other images with same face
            negative: other images with different face
            alpha: margin enforced between positive and negative pairs

    """
	with tf.variable_scope('triplet_loss'):
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
            dist_diff = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
            loss = tf.reduce_mean(tf.maximum(dist_diff, 0.0), 0)

            return loss

def batch_compute_embedding(batch):
    """
    Compute embedding from image to feature space Rd
    s.t. the squared distance between all faces of the same id is small,
    and all faces of different ids is large
    """

"""
TERMINOLOGY:
    hard positive: argmax_xpi [f(x_ai) - f(x_pi)]2,2
    hard negative: argmin_xni [f(x_ai) - f(x_ni)]2,2
    semihard negative:
        x_ni s.t. [f(x_ai) - f(x_pi)]2,2 < [f(x_ai) - f(x_ni)]2,2
"""
def _get_hard_triplet_batch(anchor, minibatch, alpha):
    """
    Compute hard positive/negatives from minibatch
    """
    distance_mat = _get_distance_matrix()

def _get_seha_triplet_batch(anchor, minibatch):

def get_triplet_batch_online(mode, minibatch, size):
"""
Online triple selection as described by the FaceNet Paper:

    Easy triplets: triplets with a loss of 0 (d(ap)+alpha < d(an))
    Hard triplets: triplets with a d(an) < d(ap)
    Semi-hard triplets: triplets where the negative is not closer to the anchor
        than the posiive, but still have non-zero loss. d(ap) < d(an) < d(ap) + alpha

    Compute argmin and argmaxes within given minibatch

"""
    if mode is "hard":
        _get_easy_triplet_batch()
    if mode is "semihard":
        _get_seha_triplet_batch()




