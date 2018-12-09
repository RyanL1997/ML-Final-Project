import discriminator.discriminator as dsc
import tensorflow as tf
import PIL


def read_image_from_path(path):
    notimg = tf.read_file(path)
    img = tf.image.decode_jpeg(notimg, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img


train_ds, test_ds = dsc.get_train_test_dataset(path="../data/lfw-deepfunneled", batchsize=100, mini=5, split_ratio=.7)
train_ds_iter = train_ds.make_one_shot_iterator()
next_el = train_ds_iter.get_next()

image = read_image_from_path(next_el)
image = tf.expand_dims(image, 0)
summary_op = tf.summary.image("image", image)

with tf.Session() as sesh:
    sesh.run(train_ds_iter.initializer)
    summary = sesh.run(summary_op)

    writer = tf.train.FileWriter('./logs')
    writer.add_summary(summary)
    writer.close()
    """
    image_input_layer = tf.get_default_graph().get_tensor_by_name("input")
    embeddings_layer = tf.get_default_graph().get_tensor_by_name("embeddings")
    phase_train = tf.get_default_graph().get_tensor_by_name("phase_train")
    global_tep = tf.get_or_create_global_step()
    embeddings, labels = dsc._get_embeddings(embeddings_layer, images, labels, \
                                                image_input_layer, \
                                                phase_train, \
                                                sess)
    """

