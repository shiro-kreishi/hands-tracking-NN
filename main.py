import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from extract_frames import extract_frames
from load_dataset import load_dataset

# Загрузка данных
# Отключите этот блок, если вы уже загрузили и обработали данные
# os.makedirs("dataset/hands_on", exist_ok=True)
# os.makedirs("dataset/hands_off", exist_ok=True)
# extract_frames("hands_on.mp4", "hands_on", "dataset/hands_on")
# extract_frames("hands_off.mp4", "hands_off", "dataset/hands_off")

# Загрузка датасета
hands_on_images, hands_on_labels = load_dataset("dataset/hands_on", 1)
hands_off_images, hands_off_labels = load_dataset("dataset/hands_off", 0)


# Изменение размера изображений на 640x480
def resize_images(images, target_size=(640, 480)):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, target_size)
        resized_images.append(resized_image)
    return np.array(resized_images)


hands_on_images_resized = resize_images(hands_on_images)
hands_off_images_resized = resize_images(hands_off_images)

X = np.concatenate((hands_on_images_resized, hands_off_images_resized), axis=0)
y = np.concatenate((hands_on_labels, hands_off_labels), axis=0)

X = X / 255.0  # Нормализация изображений
y = to_categorical(y, 2)  # One-hot кодирование меток

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Аугментация данных
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Создание модели с L2-регуляризацией и Dropout
model = Sequential([
    Conv2D(64, (5, 5), activation='relu', input_shape=(640, 480, 3), kernel_regularizer=l2(0.01)),
    MaxPooling2D((3, 3)),
    Conv2D(128, (5, 5), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((3, 3)),
    Conv2D(256, (5, 5), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели с использованием аугментации данных
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))


# Сохранение графика обучения и валидации
def plot_training_history(history, save_path='training_history.png'):
    plt.figure(figsize=(12, 6))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


plot_training_history(history)

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Сохранение модели
model.save('model.h5')
