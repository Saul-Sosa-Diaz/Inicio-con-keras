import numpy as np
import matplotlib.pyplot as plt
import keras
import os

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range 
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1) 
x_train = np.expand_dims(x_train, -1) # We have to write the new dimension to management the channel of the color
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
MODEL_PATH = "models/mnist_model.keras"

if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
else:

    shape = (28, 28, 1)
    num_classes = 10
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=shape))
    # 64: Is the number of filters (or kernels) that the layer will learn. Each filter specializes in detecting a different pattern.
    model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"))  
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) 
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(0.2)) # This avoid overfitting 
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    ]
    model.fit(x = x_train, y = y_train, batch_size = 128, epochs = 20, callbacks=callbacks)
    score = model.evaluate(x_test, y_test, verbose = 0)
    model.save(MODEL_PATH)

predictions = model.predict(x_test)

for index in range(len(x_test)):
    predicted = np.argmax(predictions[index])
    print(f"\nThe model predicts: {predicted}")
    real_label = y_test[index]
    print(f"The real label of the image is: {real_label}")
    plt.imshow(x_test[index].squeeze(), cmap=plt.cm.binary)
    plt.title(f"Prediction: {predicted} | Real: {real_label}")
    plt.show()

