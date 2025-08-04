from tensorflow import keras
# from tensorflow.keras import layers, models
from utils.preprocessing import create_train_test_split
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras import layers, models

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

data_dir = "data/raw"  #Kaggle dataset
X_train, X_test, y_train, y_test = create_train_test_split(data_dir)

# Model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test))

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/fabric_defect.h5")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('training_history.png')
plt.show()