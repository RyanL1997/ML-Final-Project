import argparse
import os
import requests
import zipfile
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from sklearn.svm import SVC

model_dict = {
    '20170511-185253': '0B5MzpY9kBtDVOTVnU3NIaUdySFE'
}


def download_and_extract_model(model_name, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_id = model_dict[model_name]
    destination = os.path.join(data_dir, model_name + '.zip')
    if not os.path.exists(destination):
        print('Downloading model to %s ' % destination)
        download_file_from_google_drive(file_id, destination)

        with zipfile.ZipFile(destination, 'r') as zip_ref:
            print('Extracting model to %s ' % data_dir)
            zip_ref.extractall(data_dir)


def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for k, v in response.cookies.items():
        if k.startswith('download_warning'):
            return v

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def read_data(image_paths, label_list, image_size, batch_size,
              max_nrof_epochs, num_threads):
    images = ops.convert_to_tensor(image_paths, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices(tuple(images, labels))\
        .shuffle(image_size).repeat(max_nrof_epochs)
    images_labels = []
    imgs = []
    lbls = []
    for _ in range(num_threads):
        image, label = read_image_from_disk(filename_to_label_tuple=dataset)
        image = tf.random_crop(image, size=[image_size, image_size, 3])
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_standardization(image)

        imgs.append(image)
        lbls.append(label)
        images_labels.append([image, label])
    image_batch, label_batch = dataset.interleave().batch(batch_size)
    return image_batch, label_batch


def read_image_from_disk(filename_to_label_tuple):
    label = filename_to_label_tuple[1]
    file_contents = tf.read_file(filename_to_label_tuple[0])
    img = tf.image.decode_jpeg(file_contents, channels=3)
    return img, label


def _load_model(model_filepath):
    model_expand = os.path.expanduser(model_filepath)
    if os.path.isfile(model_expand):
        with gfile.FastGFile(model_expand, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')


def _create_embeddings(embedding_layer, images, labels, images_placeholder,
                       phase_train_placeholder, sess):
    embs = None
    lbls = None

    try:
        i = 0
        while True:
            batch_images, batch_labels = sess.run([images, labels])
            emb = sess.run(embedding_layer,
                           feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})
            embs = np.concatenate([embs, emb]) if embs is not None else emb
            lbls = np.concatenate([lbls, batch_labels]) if lbls is not None else batch_labels
            i += 1

    except tf.errors.OutOfRangeError:
        pass

    return embs, lbls


def main(input_image_dir, model_path, classifier_storage, batch_size, num_threads, num_epochs,
         min_imgs_per_label, split_ratio, is_train=True):

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        train_set, test_set = _get_test_and_train_set(input_image_dir,train_test,
                                                       min_imgs_per_label=5, split_ratio=split_ratio)

        if is_train:
            images, labels, class_names = _load_images_and_labels(train_set, image_size=160, batch_size=batch_size,
                                                                  num_threads=num_threads, num_epochs=num_epochs)

        _load_model(model_filepath=model_path)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        embs, lbls = _create_embeddings(embedding_layer, images, labels,
                                        images_placeholder, phase_train_placeholder, sess)

        if is_train:
            _train_and_save(embs, lbls, class_names, classifier_storage)


def _train_and_save(embs, lbls, class_names, classifier_storage):
    model = SVC(kernel='linear', probability=True, verbose=False)
    model.fit(embs, lbls)

    with open(classifier_storage, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)


def _get_test_and_train_set(in_dir, min_imgs_per_label, split_ratio):
    dataset = get_dataset(in_dir)
    dataset = filter_dataset(dataset, min_imgs_per_label)
    train_set, test_set = split_dataset(dataset, split_ratio=split_ratio)
    return train_set, test_set


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(int(len(dataset))):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def _load_images_and_labels(dataset, image_size, batch_size, num_threads, num_epochs):
    class_names = [cls.name for cls in dataset]
    image_paths, labels = get_image_paths_and_labels(dataset)
    images, labels = read_data(image_paths, labels, image_size, batch_size, num_epochs, num_threads)

    return images, labels, class_names


def get_dataset(in_dir):
    dataset = []


    classes = os.listdir(in_dir)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        class_dir = os.path.join(in_dir, class_name)
        if os.path.isdir(class_dir):
            images = os.listdir(class_dir)
            image_paths = [os.path.join(class_dir, img) for img in images]
            dataset.append(ImageSet(class_name, image_paths))

    return dataset


def filter_dataset(dataset, min_imgs_per_label):
    filtered = []
    for i in range(len(dataset)):
        if len(dataset[i].image_paths) < min_imgs_per_label:
            continue
        else:
            filtered.append(dataset[i])
    return filtered


def split_dataset(dataset, split_ratio):
    train_set = []
    test_set = []
    min_nrof_images = 2
    for clss in dataset:
        paths = clss.image_paths
        np.random.shuffle(paths)
        split = int(round(len(paths) * split_ratio))
        if split < min_nrof_images:
            continue
        train_set.append(ImageSet(clss.name, paths[0:split]))
        test_set.append(ImageSet(clss.name, paths[split:-1]))

    return train_set, test_set


class ImageSet():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model-path', type=str, action='store', dest='model_path',
                        help='Path to model protobuf graph')
    parser.add_argument('--input-dir', type=str, action='store', dest='input_dir',
                        help='Path to data train on')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size',
                        help='Size of batch')
    parser.add_argument('--num-threads', type=int, action='store', dest='num_threads', default=16,
                        help='Number of threads')
    parser.add_argument('--num-epochs', type=int, action='store', dest='num_epochs', default=3,
                        help="Number of epochs")
    parser.add_argument('--min-num-images-per-class', type=int, action='store', default=10,
                        dest='min_image_per_class', help="Minimum number of images per class")
    parser.add_argument('--classifier-path', type=str, action='store', dest='classifier_path',
                        help='Path to output trained classifier model')
    parser.add_argument('--is-train', action='store_true', dest='is_train', default=False,
                        help='Flag to determine if train or evaluate')

    args = parser.parse_args()

    main(input_directory, args.input_dir, model_path=args.model_path,
         classifier_storage=args.classifier_path, batch_size=args.batch_size, num_threads=args.num_threads,
         num_epochs=args.num_epochs, min_imgs_per_label=args.min_images_per_class, split_ratio=args.split_ratio,
         is_train=args.is_train)