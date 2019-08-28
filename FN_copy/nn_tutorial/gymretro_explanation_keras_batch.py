import numpy as np
from tensorflow.python import keras as K
from collections import deque,namedtuple
import retro
# 2-layer neural network.
# model = K.Sequential([
#     K.layers.Dense(units=4, input_shape=((2, )),
#                    activation="sigmoid"),
#     K.layers.Dense(units=4),
# ])
Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])
buffer_size=1024
experiences = deque(maxlen=buffer_size)

env = retro.make(game='Pong-Atari2600', players=1)
s = env.reset()

a = env.action_space.sample()
n_state, reward, done, info = env.step(a)

e = Experience(s, a, reward, n_state, done)
experiences.append(e)

feature_shape = experiences[0].s.shape

normal = K.initializers.glorot_normal()
model = K.Sequential()
model.add(K.layers.Conv2D(
    32, kernel_size=8, strides=4, padding="same",
    input_shape=feature_shape, kernel_initializer=normal,
    activation="relu"))
model.add(K.layers.Conv2D(
    64, kernel_size=4, strides=2, padding="same",
    kernel_initializer=normal,
    activation="relu"))
model.add(K.layers.Conv2D(
    64, kernel_size=3, strides=1, padding="same",
    kernel_initializer=normal,
    activation="relu"))
model.add(K.layers.Flatten())
model.add(K.layers.Dense(256, kernel_initializer=normal,
                            activation="relu"))
model.add(K.layers.Dense(env.action_space.n,
                            kernel_initializer=normal))


print(feature_shape)
# Make batch size = 3 data (dimension of x is 2).
batch = np.random.rand(1,210,160,3)

y = model.predict(batch)
print(y.shape)  # Will be (1, 8)
print(y)
print(y.argmax())

index=int(y.argmax())
array=np.zeros(env.action_space.n,dtype=int)
array[index]=1
print(array)