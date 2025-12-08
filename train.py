import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 48
BATCH = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH,
    shuffle=True)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH,
    shuffle=True)

norm = layers.Rescaling(1./255) # Normalize pixel values to [0, 1]
train_ds = train_ds.map(lambda x, y: (norm(x), y)) #x=images, y=labels
test_ds = test_ds.map(lambda x, y: (norm(x), y))

model = models.Sequential([
    layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(48,48,1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'), #64 filters of 3x3
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'), 
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=test_ds, epochs=20)

model.save("emotion_model.h5")
print("Model saved as emotion_model.h5")