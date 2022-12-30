import tensorflow as tf
import matplotlib.pyplot as plt

# Load the data
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Check that it's actuall images
plt.imshow(x_train[0], cmap=plt.cm.binary)

# Normalize data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create model
model = tf.keras.models.Sequential()

# Add layers
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

# Layer
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

# Layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, shuffle=True, epochs=100)

'''
I use a MacBook Pro, so I stopped the training after 4 epochs (100 epochs was going to take around 6 hours)
Epoch 1/100
50000/50000 [==============================] - 221s 4ms/step - loss: 1.8571 - acc: 0.3043
Epoch 2/100
50000/50000 [==============================] - 246s 5ms/step - loss: 1.4137 - acc: 0.4910
Epoch 3/100
50000/50000 [==============================] - 236s 5ms/step - loss: 1.1943 - acc: 0.5744
Epoch 4/100
50000/50000 [==============================] - 238s 5ms/step - loss: 1.0693 - acc: 0.6186
Epoch 5/100
 7296/50000 [===>..........................] - ETA: 3:31 - loss: 1.0180 - acc: 0.6419
'''

# Evaluate model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)