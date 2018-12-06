import discriminator as dsc
import tensorflow as tf

dsc.get_dataset("../data/faces/lfw", 100)
tensor = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sesh:
    print(sesh.run(tensor))

