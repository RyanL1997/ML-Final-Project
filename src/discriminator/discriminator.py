import tensorflow as tf
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
        classes = [path for path in os.listdir(path) if os.path.isdir(os.path.join(base_path, path))]
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
        test_dataset = test_dataset.prefetch(1)
        return train_dataset, test_dataset


def _resize_images(path, label):
    """
    To be mapped to dataset.
    Resizes image specified by path.
    """
    notimg = tf.read_file(path)
    img = tf.image.decode_jpeg(notimg, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize_images(img, [64, 64])
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

"""
def train(batch):
    with tf.Session() as sess:
        global_step = tf.Variable(0, trainable=False)

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input")
        labels_placeholder = tf.get_default_graph().get_tensoy_by_name("")
        phase_train = tf.get_default_graph().get_tensor_by_name("phase_train")
        embeddings_layers = tf.get_default_graph().get_tensor_by_name("embeddings")

        embeddings, labels = _get_embeddings(embeddings_layers, images_placeholder, images, labels_placeholder, phase_train, sess)
"""

def _get_embeddings(embedding_layer, images_placeholder, images, labels,  phase_train_placeholder, sess):
    emb_array = None
    label_array = None
    try:
        i = 0
        while True:
            batch_images, batch_labels = sess.run([images, labels])
            # logger.info('Processing iteration {} batch of size: {}'.format(i, len(batch_labels)))
            emb = sess.run(embedding_layer,
                           feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})

            emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
            label_array = np.concatenate([label_array, batch_labels]) if label_array is not None else batch_labels
            i += 1

    except tf.errors.OutOfRangeError:
        pass
    return emb_array, label_array


def get_distance_matrix(batch):
    batch_squared = tf.matmul(batch, tf.transpose(batch))
    squares = tf.diag_part(batch_squared)
    a2 = tf.expand_dims(squares, 0)
    b2 = tf.expand_dims(squares, 1)
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



def get_triplet_batch_online(mode, minibatch, size):
    """
    TERMINOLOGY:
        hard positive: argmax_xpi [f(x_ai) - f(x_pi)]2,2
        hard negative: argmin_xni [f(x_ai) - f(x_ni)]2,2
        semihard negative:
            x_ni s.t. [f(x_ai) - f(x_pi)]2,2 < [f(x_ai) - f(x_ni)]2,2
    Online triple selection as described by the FaceNet Paper:

        Easy triplets: triplets with a loss of 0 (d(ap)+alpha < d(an))
        Hard triplets: triplets with a d(an) < d(ap)
        Semi-hard triplets: triplets where the negative is not closer to the anchor
            than the posiive, but still have non-zero loss. d(ap) < d(an) < d(ap) + alpha

        Compute argmin and argmaxes within given minibatch
    """
