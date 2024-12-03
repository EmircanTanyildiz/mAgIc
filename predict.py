import numpy as np
import os
import cv2
import random
import tensorflow as tf
from flask import Flask, request, jsonify


app = Flask(__name__)


from tensorflow.keras.losses import SparseCategoricalCrossentropy

class CustomSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):
    def __init__(self, **kwargs):
        kwargs.pop("fn", None)  
        if kwargs.get("reduction") == "auto":
            kwargs["reduction"] = "sum_over_batch_size"  
        super().__init__(**kwargs)


model_path = "D:\AIProjectAPI\model_1.h5"
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"SparseCategoricalCrossentropy": CustomSparseCategoricalCrossentropy}
)


def format_frames(frame, output_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
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


categories = ["Standing Still", "Meet and Split", "Walking", "Sitting", "Clapping"]


@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json()
    if not data or 'video_path' not in data:
        return jsonify({"error": "No video_path provided"}), 400

    video_file_path = data['video_path']

    if not os.path.exists(video_file_path):
        return jsonify({"error": f"File {video_file_path} does not exist"}), 404

    try:
        n_frames = 10  
        processed_video = frames_from_video_file(video_file_path, n_frames=n_frames)
        processed_video = np.expand_dims(processed_video, axis=0)  # Batch boyutu eklenir

       
        predictions = model.predict(processed_video)

       
        predicted_class = np.argmax(predictions)
        predicted_label = categories[predicted_class]
        print(predicted_label)
        return jsonify({"predicted_class": predicted_label}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
