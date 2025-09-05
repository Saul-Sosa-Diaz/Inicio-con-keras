import os
import tensorflow as tf
import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

BASE_FOLDER_PATH = "data/PetImages"
MODEL_PATH = "models/cat_vs_dog_model.keras"
image_size = (180, 180)
print("================ 1: Cleaning ================")
num_deleted = 0

for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(BASE_FOLDER_PATH, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()
        except (IOError, SyntaxError) as e:
            num_deleted += 1
            os.remove(fpath)

print(f"\nCleared: {num_deleted} corrupted images.")
print("======================================================================")


print("\n==========2: Load data ==========")

batch_size = 128

train, val = keras.utils.image_dataset_from_directory(
    BASE_FOLDER_PATH,
    validation_split=0.2,
    subset="both",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
)
print("======================================================================")

print("\n============3:Show images ============")
plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
    for i in range(9):
        if i < len(images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
plt.show()
print("======================================================================")


print("\n============5:augmentation data ============")
data_augmentation_layers = [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


plt.figure(figsize=(10, 10))
for images, _ in train.take(1):
    for i in range(9):
        if i < len(images):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
plt.show()


print("\n============ 5: Build model ============")
input_shape = (128, 128, 3)
inputs = keras.Input(shape=input_shape)


# def make_model(input_shape, num_classes):
#     inputs = keras.Input(shape=input_shape)
#     # Entry block
#     x = keras.layers.Rescaling(1.0 / 255)(inputs)
#     x = keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)

#     previous_block_activation = x  # Set aside residual

#     for size in [256, 512, 728]:
#         x = keras.layers.Activation("relu")(x)
#         x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = keras.layers.BatchNormalization()(x)

#         x = keras.layers.Activation("relu")(x)
#         x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = keras.layers.BatchNormalization()(x)

#         x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

#         # Project residual
#         residual = keras.layers.Conv2D(size, 1, strides=2, padding="same")(
#             previous_block_activation
#         )
#         x = keras.layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual

#     x = keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)

#     x = keras.layers.GlobalAveragePooling2D()(x)
#     if num_classes == 2:
#         units = 1
#     else:
#         units = num_classes

#     x = keras.layers.Dropout(0.25)(x)
#     # We specify activation=None so as to return logits
#     outputs = keras.layers.Dense(units, activation=None)(x)
#     return keras.Model(inputs, outputs)


# model = make_model(input_shape=image_size + (3,), num_classes=2)


model = keras.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(keras.layers.Rescaling(1.0 / 255))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dropout(0.2))  # This avoid overfitting
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(
    train,
    epochs=20,
    validation_data=val,
)
model.save(MODEL_PATH)

model = keras.models.load_model(MODEL_PATH)
img = keras.utils.load_img(
    "data/tests/pets/Upside_down_gray_cat.jpg", target_size=image_size
)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

img = keras.utils.load_img(
    "data/tests/pets/gato2.jpg", target_size=image_size
)
plt.imshow(img)
plt.show()

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

img = keras.utils.load_img("data/tests/pets/gato3.jpg", target_size=image_size)
plt.imshow(img)
plt.show()

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

img = keras.utils.load_img("data/tests/pets/dog.jpg", target_size=image_size)
plt.imshow(img)
plt.show()

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")