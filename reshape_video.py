import cv2

# Открываем видео файл
cap = cv2.VideoCapture('video.mp4')

# Получаем исходные размеры видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Устанавливаем новые размеры
new_width = 640
new_height = 480

# Создаем видео-писатель
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('hands_off.mp4', fourcc, 1.0, (new_width, new_height))

while True:
    # Читаем кадр
    ret, frame = cap.read()

    if ret:
        # Обрезаем кадр сверху на 200 пикселей
        cropped_frame = frame[400:height, 0:width]

        # Изменяем размер обрезанного кадра
        resized_frame = cv2.resize(cropped_frame, (new_width, new_height))

        # Записываем обрезанный и измененный кадр
        out.write(resized_frame)

        # Отображаем видео
        cv2.imshow('Video', resized_frame)

        # Ожидаем нажатия клавиши 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()
