#import datascience and AI tools
import keras
import tensorflow
import numpy as np

#set render and processing range (What you will be processing)
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#set optimizer as sgd
model.compile(optimizer='sgd', loss = 'mean_squared_error')

#create arrays
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0], dtype=float)

#specify what render (range was specified on line 6 and 7
#specify epochs (iterations)
model.fit(xs, ys, epochs=1000)

#print result
print(model.predict([7.0]))

#jinoo
print('In honor of JINOO!')