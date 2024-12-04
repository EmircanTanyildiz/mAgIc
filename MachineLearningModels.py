import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATASET_PATH = "D:/AI_Project_Beykoz_Uni_2204040177/Human Activity Recognition - Video Dataset"


def extract_optical_flow_opencl(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    features = []
    ret, prev_frame = cap.read()
    if not ret:
        return None
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    

    prev_frame_gpu = cv2.UMat(prev_frame)

    while len(features) < num_frames:
        ret, next_frame = cap.read()
        if not ret:
            break
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        

        next_frame_gpu = cv2.UMat(next_frame)

       
        flow_gpu = cv2.calcOpticalFlowFarneback(prev_frame_gpu, next_frame_gpu, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        
        flow = flow_gpu.get()
        
        
        features.append(flow.mean(axis=(0, 1)))
        prev_frame_gpu = next_frame_gpu

    cap.release()
    if len(features) < num_frames:
        return None
    return np.array(features).flatten()


def load_data(dataset_path):
    features = []
    labels = []

    for label_folder in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label_folder)
        if os.path.isdir(label_path):
            for video_file in os.listdir(label_path):
                video_path = os.path.join(label_path, video_file)
                feature = extract_optical_flow_opencl(video_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(label_folder)

    return np.array(features), np.array(labels)


X, y = load_data(DATASET_PATH)
print(f"Özellik boyutu: {X.shape}, Sınıf sayısı: {len(np.unique(y))}")


le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=300),
    "KNN": KNeighborsClassifier(n_neighbors=55),
}


results = {}

for name, model in models.items():
    print(f"Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

  
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

 
    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }


metrics = ["accuracy", "precision", "recall", "f1_score"]
model_names = list(results.keys())
scores = {metric: [results[model][metric] for model in model_names] for metric in metrics}


for metric, values in scores.items():
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, values, color='skyblue')
    plt.title(f"Model Performansı - {metric.capitalize()}")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Modeller")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.show()


for name, result in results.items():
    cm = result["confusion_matrix"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
