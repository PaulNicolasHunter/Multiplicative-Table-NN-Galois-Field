# import matplotlib.pyplot as plt
# from itertools import permutations
# i = list(permutations(range(9), 2))
# plt.plot(i, )

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from tensorflow import keras as ks

logits = np.array([
	[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 0],
	[0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1],
	[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 0],
	[0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1],
	[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 0],
	[0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 1],
	[0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0],
	[0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1],
	[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1], [1, 0, 0, 1, 0, 0],
	[1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1],
	[1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0],
	[1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1],
	[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1], [1, 1, 0, 1, 0, 0],
	[1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1],
	[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0],
	[1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]).astype(dtype=np.float16).reshape(64, 6)

labels = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
				   [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
				   [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1],
				   [0, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0],
				   [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0],
				   [0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1],
				   [0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1],
				   [0, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]]).astype(
	dtype=np.float16).reshape(64, -1, 3)

layers = [
	ks.layers.LSTM(6, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
	kernel_initializer = 'glorot_uniform', recurrent_initializer = 'orthogonal', bias_initializer = 'zeros')
]

model = ks.models.Sequential(layers)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

for _ in range(64):
	model.fit(logits[_].reshape(-1, 6), labels[_], epochs=15)

predictions_ = np.array([])
for _ in logits:
	predictions_ = np.append(predictions_, model.predict(_.reshape(-1, 6)))

# for _ in predictions_.reshape(-1):
# 	_ = tf.math.round(_)

print(predictions_, np.around(predictions_).reshape(-1, 3))
