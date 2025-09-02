
```python
import keras

# 1. Define the structure of model
model = keras.Sequential([
    layers.Input(shape=(784,)),          
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax") 
])

# 2. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Train the model
# x_train: training data, y_train: training labels
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2 # Use 20% of the data for validation
)

# 4. Evaluate the model
# x_test: test data, y_test: test labels
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy on the test set: {accuracy:.4f}")

# 5. Make predictions
predictions = model.predict(x_new_data)
```

---
## Layers Guide(`keras.layers`)
Layers are the building blocks of your network. Here are the most important ones and when to use them.

### Essential Layers (For all model types)

| layer | What It Does | When to Use It | Key Parameters |
| :--- | :--- | :--- | :--- |
| **`Input`** | Defines the shape of the input data. | **Always** as the first layer of the model. | `shape=(...)` |
| **`Dense`** | Standard fully connected neural layer. | Base layer for most tasks. Ideal for output and classification layers. | `units` (number of neurons), `activation` |
| **`Dropout`** | Randomly "turns off" neurons during training. | To **reduce overfitting**. Usually placed after dense or convolutional layers. | `rate` (e.g., 0.5 to turn off 50%) |
| **`Flatten`** | Flattens multi-dimensional data to 1D vector. | To connect convolutional layers to dense layers. (e.g., from `(28, 28, 1)` to `(784,)`) | N/A |

### Convolutional Layers (For Computer Vision)

| layer | What It Does | When to Use It | Key Parameters |
| :--- | :--- | :--- | :--- |
| **`Conv2D`** | Detects local patterns (edges, textures) in images. | The **core of CNNs** for any computer vision task. | `filters`, `kernel_size`, `activation` |
| **`MaxPooling2D`**| Reduces the image size (downsampling) by taking the maximum value. | After `Conv2D` layers to reduce computational load and make the model more robust. | `pool_size` (e.g., `(2, 2)`) |
| **`GlobalAveragePooling2D`** | Reduces each feature map to a single number (its average). | Modern alternative to `Flatten` to connect convolutional layers to the output `Dense` layer. Helps prevent overfitting. | N/A |

### Capas Recurrentes (Para Datos Secuenciales: Texto, Series Temporales)

| Layer | What It Does | When to Use It | Key Parameters |
| :--- | :--- | :--- | :--- |
| **`Embedding`** | Converts integers (e.g., word indices) into dense fixed-size vectors. | **Always** as the first layer in Natural Language Processing (NLP) models. | `input_dim` (vocabulary size), `output_dim` (vector size) |
| **`LSTM`** | Advanced recurrent network, capable of learning long-term dependencies. | The standard for most sequence tasks (NLP, time series). Solves the vanishing gradient problem. | `units` (number of neurons) |
| **`GRU`** | Similar to LSTM but simpler and computationally faster. | Good alternative to `LSTM`, especially if training speed is a priority. | `units` (number of neurons) |
| **`SimpleRNN`**| The most basic recurrent layer. | For learning very short sequences or educational purposes. In practice, `LSTM` or `GRU` are almost always better. | `units` (number of neurons) |

### Funciones de Activación (`activation=`)

Used as a parameter in many layers to introduce non-linearity.

*   **`"relu"`**: The most popular for hidden layers. Fast and efficient.
*   **`"sigmoid"`**: Compresses values between 0 and 1. Ideal for the output layer in **binary classification**.
*   **`"softmax"`**: Converts outputs into a probability vector that sums to 1. Ideal for the output layer in **multiclass classification**.
*   **`"tanh"`**: Compresses values between -1 and 1.

---
## Guía de compilación
## Compilation Guide
### Optimizer (`optimizer`)
The algorithm that updates the network weights to minimize the loss.
*   **`"adam"`**: The best to start with. Efficient and works well in most cases.
*   **`"rmsprop"`**: Good for recurrent models (RNNs).
*   **`"sgd"`**: Stochastic Gradient Descent. Classic, but often slower to converge.

### Loss Function (`loss`)
Measures how accurate the model is during training. The choice is CRITICAL and depends on your problem.
*   **Binary Classification (2 classes):**
    *   `"binary_crossentropy"`
*   **Multiclass Classification (>2 classes):**
    *   `"categorical_crossentropy"`: If your labels are in *one-hot* format (e.g., `[0, 1, 0, 0]`).
    *   `"sparse_categorical_crossentropy"`: If your labels are integers (e.g., `1`). **(Most common)**.
*   **Regression (predicting a numeric value):**
    *   `"mean_squared_error"` (MSE) or `"mse"`

### Metrics (`metrics`)
Functions used to monitor model performance.
*   **`["accuracy"]`**: The most common for classification. Measures the percentage of correct predictions.
*   **`["mae"]`**: Mean Absolute Error. Useful for regression.

---
