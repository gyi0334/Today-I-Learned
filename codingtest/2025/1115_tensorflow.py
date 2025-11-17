import tensorflow as tf
import numpy as np

tf.random.set_seed(1)
t_x = tf.random.uniform([4,3], dtype=tf.float32)
t_y = tf.range(4)
print(t_x, t_y)

ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)
print(ds_x, ds_y)

ds_joint = tf.data.Dataset.zip((ds_x, ds_y))
