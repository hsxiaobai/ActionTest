import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(height=32, width=32),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(256, activation="relu"),
    
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="nadam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

print("Start Training")
model.fit(x_train, y_train,     epochs=10)
print("Training Finished")

print("Start Evaluating")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Evaluating Finished:\nLoss: {loss} Accuracy: {accuracy}")

model.save_weights("cifar10", save_format="tf")
