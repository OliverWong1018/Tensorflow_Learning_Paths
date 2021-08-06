import tensorflow as tf
import numpy as np
with tf.device("cpu"):
    a = tf.constant(1.)
print (a)
print(type(a))
print(a.shape)
print(a.ndim)
print(a.numpy())
print(tf.is_tensor(a))
print("------------------")

b = np.ones([2,3])
print(b)
print(type(b))
print(b.shape)
c = tf.convert_to_tensor(b)
print(c)
print(c.dtype)
print(c.shape)
print(c.numpy())
print("------------------")

#正态分布
d1 = tf.random.normal([2,2],mean =1, stddev= 1)
print(d1)
#
d2 = tf.random.truncated_normal([2,2], mean=0, stddev=1)
print(d2)
#均匀分布
d3 = tf.random.uniform([2,2],minval =0, maxval=1)
print(d3)

out = tf.random.uniform([4,10])
print(out)

y = tf.range(4)
y = tf.one_hot(y,depth=10)
print(y)

loss= tf.keras.losses.mae(y,out)
print(loss)
loss= tf.reduce_mean(loss)
print(loss)
print("------------------")


test = tf.ones([1,2,3,4])
print(test)
print(test[0][0])