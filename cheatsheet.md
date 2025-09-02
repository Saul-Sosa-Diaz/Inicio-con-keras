# Cheatsheet Keras

## Ejemplo:
```python
import keras
from keras import layers

# 1. Definir la arquitectura del modelo
model = keras.Sequential([
    layers.Input(shape=(784,)),          
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2), # Esto evita el overfitting
    layers.Dense(10, activation="softmax") 
])

# 2. Compilar el modelo
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Entrenar el modelo
# x_train: datos de entrenamiento, y_train: etiquetas de entrenamiento
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2 # Usa el 20% de los datos para validación
)

# 4. Evaluar el modelo
# x_test: datos de prueba, y_test: etiquetas de prueba
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

# 5. Realizar predicciones
predictions = model.predict(x_new_data)
```

---
## Guía de Capas (`keras.layers`)

Las capas son los bloques de construcción de tu red. Aquí están las más importantes y cuándo usarlas.

### Capas Esenciales (Para todo tipo de modelo)

| Capa | Qué Hace | Cuándo Usarla | Parámetros Clave |
| :--- | :--- | :--- | :--- |
| **`Input`** | Define la forma de los datos de entrada. | **Siempre** como la primera capa del modelo. | `shape=(...)` |
| **`Dense`** | Capa neuronal estándar, totalmente conectada. | Capa base para la mayoría de tareas. Ideal para capas de salida y clasificación. | `units` (nº de neuronas), `activation` |
| **`Dropout`** | "Apaga" neuronas aleatoriamente durante el entrenamiento. | Para **reducir el sobreajuste (overfitting)**. Se suele poner después de capas densas o convolucionales. | `rate` (ej: 0.5 para apagar el 50%) |
| **`Flatten`** | Aplana los datos de múltiples dimensiones a un vector 1D. | Para conectar capas convolucionales con capas densas. (ej: de `(28, 28, 1)` a `(784,)`) | N/A |

### Capas Convolucionales (Para Visión por Computadora)

| Capa | Qué Hace | Cuándo Usarla | Parámetros Clave |
| :--- | :--- | :--- | :--- |
| **`Conv2D`** | Detecta patrones locales (bordes, texturas) en imágenes. | El **corazón de las CNNs** para cualquier tarea de visión por computadora. | `filters`, `kernel_size`, `activation` |
| **`MaxPooling2D`**| Reduce el tamaño de la imagen (downsampling) tomando el valor máximo. | Después de las capas `Conv2D` para reducir la carga computacional y hacer el modelo más robusto. | `pool_size` (ej: `(2, 2)`) |
| **`GlobalAveragePooling2D`** | Reduce cada mapa de características a un solo número (su promedio). | Alternativa moderna a `Flatten` para conectar capas convolucionales a la capa de salida `Dense`. Ayuda a prevenir el sobreajuste. | N/A |

### Capas Recurrentes (Para Datos Secuenciales: Texto, Series Temporales)

| Capa | Qué Hace | Cuándo Usarla | Parámetros Clave |
| :--- | :--- | :--- | :--- |
| **`Embedding`** | Convierte enteros (ej: índices de palabras) en vectores densos de tamaño fijo. | **Siempre** como primera capa en modelos de Procesamiento de Lenguaje Natural (NLP). | `input_dim` (tamaño del vocabulario), `output_dim` (tamaño del vector) |
| **`LSTM`** | Red Recurrente avanzada, capaz de aprender dependencias a largo plazo. | El estándar para la mayoría de tareas con secuencias (NLP, series temporales). Resuelve el problema de desvanecimiento de gradiente. | `units` (nº de neuronas) |
| **`GRU`** | Similar a LSTM pero más simple y computacionalmente más rápida. | Buena alternativa a `LSTM`, especialmente si la velocidad de entrenamiento es una prioridad. | `units` (nº de neuronas) |
| **`SimpleRNN`**| La capa recurrente más básica. | Para aprender secuencias muy cortas o con fines educativos. En la práctica, `LSTM` o `GRU` son casi siempre mejores. | `units` (nº de neuronas) |

### Funciones de Activación (`activation=`)

Se usan como parámetro en muchas capas para introducir no-linealidad.

*   **`"relu"`**: La más popular para capas ocultas. Rápida y eficiente.
*   **`"sigmoid"`**: Comprime los valores entre 0 y 1. Ideal para la capa de salida en **clasificación binaria**.
*   **`"softmax"`**: Convierte las salidas en un vector de probabilidades que suma 1. Ideal para la capa de salida en **clasificación multiclase**.
*   **`"tanh"`**: Comprime los valores entre -1 y 1.

---
## Guía de compilación
### Optimizador (`optimizer`)
El algoritmo que actualiza los pesos de la red para minimizar la pérdida.
*   **`"adam"`**: El mejor para empezar. Es eficiente y funciona bien en la mayoría de los casos.
*   **`"rmsprop"`**: Bueno para modelos recurrentes (RNNs).
*   **`"sgd"`**: Descenso de Gradiente Estocástico. Clásico, pero a menudo más lento para converger.

### Función de Pérdida (`loss`)
Mide qué tan acertado es el modelo durante el entrenamiento. La elección es CRÍTICA y depende de tu problema.
*   **Clasificación Binaria (2 clases):**
    *   `"binary_crossentropy"`
*   **Clasificación Multiclase (>2 clases):**
    *   `"categorical_crossentropy"`: Si tus etiquetas están en formato *one-hot* (ej: `[0, 1, 0, 0]`).
    *   `"sparse_categorical_crossentropy"`: Si tus etiquetas son enteros (ej: `1`). **(Más común)**.
*   **Regresión (predecir un valor numérico):**
    *   `"mean_squared_error"` (MSE) o `"mse"`

### Métricas (`metrics`)
Funciones que se usan para monitorear el rendimiento del modelo.
*   **`["accuracy"]`**: La más común para clasificación. Mide el porcentaje de predicciones correctas.
*   **`["mae"]`**: Error Absoluto Medio. Útil para regresión.

---
