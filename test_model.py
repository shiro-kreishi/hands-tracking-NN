import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Загрузка обученной модели
model = load_model('model.h5')

# Получение размеров модели
model_input_shape = (640, 480)

# Инициализация камеры
cap = cv2.VideoCapture(0)  # 0 для встроенной веб-камеры, или выберите другое устройство

# Установка разрешения видео
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обрезка и изменение размера кадра
    frame = cv2.resize(frame, model_input_shape)

    # Предобработка кадра
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = img_to_array(frame)
    frame = preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)

    # Предсказание с использованием модели
    prediction = model.predict(frame)
    if prediction[0][1] > 0.5:
        status = "Hands On Keyboard"
    else:
        status = "Hands Off Keyboard"

    # Вывод результата на экран
    frame = cv2.cvtColor(frame[0], cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (640, 480))
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Status', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
