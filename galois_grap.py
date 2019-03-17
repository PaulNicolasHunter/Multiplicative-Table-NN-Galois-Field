# import matplotlib.pyplot as plt
# from itertools import permutations
# i = list(permutations(range(9), 2))
# plt.plot(i, )
import tensorflow as tf

inputs = tf.placeholder([6])
outs = tf.placeholder([50, 3])

h1 = tf.constant(6, 50)
h2 = tf.constant(50, 70)
h3 = tf.constant(70, 50)

a = tf.Variable(tf.truncated_normal())