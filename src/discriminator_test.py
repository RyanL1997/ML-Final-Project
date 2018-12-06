import discriminator as dsc
import tensorflow as tf

train_ds, test_ds = dsc.get_train_test_dataset("../data/lfw-deepfunneled", 100, 5, .7)
# train_ds_iter = train_ds.make_one_shot_iterator()
# next_el = train_ds_iter.get_next()

with tf.Session() as sesh:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # sess.run(train_ds_iter.initializer)
    # sess.run(init_op)

    tf.print(train_ds.output_classes)
    tf.print(train_ds.output_shapes)
    tf.print(train_ds.output_types)

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

