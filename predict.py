import numpy as np
import os
import cv2
import random
import tensorflow as tf

# Model kayıp fonksiyonu için özel bir sınıf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class CustomSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):
    def __init__(self, **kwargs):
        kwargs.pop("fn", None)  # 'fn' parametresini kaldır
        if kwargs.get("reduction") == "auto":
            kwargs["reduction"] = "sum_over_batch_size"  # Desteklenen değere güncelle
        super().__init__(**kwargs)

# Modeli yükle
model_path = "D:/AI_Project_Beykoz_Uni_2204040177/model (1).h5"
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"SparseCategoricalCrossentropy": CustomSparseCategoricalCrossentropy}
)

# Videodan kareleri işlemek için yardımcı fonksiyonlar
def format_frames(frame, output_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB'ye çevir
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    ret, frame = src.read()
    if not ret:
        print(f"Error reading frame from {video_path}")
        return np.zeros((n_frames, *output_size, 3))

    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))

    src.release()
    result = np.array(result)
    return result

# Video tahminleme
video_path = "D:/AI_Project_Beykoz_Uni_2204040177/Human Activity Recognition - Video Dataset"
categories = []
categories.append("Standing Still")
categories.append("Meet and Split")
categories.append("Walking")
categories.append("Sitting")
categories.append("Clapping")

categories = list(categories)
for i in categories:
    print(i)
n_frames = 10  # Modelin eğitiminde kullanılan kare sayısı

# İşlenecek video yolu
video_file_path ="D:\AI_Project_Beykoz_Uni_2204040177\Human Activity Recognition - Video Dataset\Sitting\Sitting (4).mp4"

processed_video = frames_from_video_file(video_file_path, n_frames=n_frames)
processed_video = np.expand_dims(processed_video, axis=0)  # Batch boyutu eklenir

# Tahmin yapın
predictions = model.predict(processed_video)

# Sınıf tahminini alın
predicted_class = np.argmax(predictions)
predicted_label = categories[predicted_class]

print(f"Predicted class: {predicted_label}")
