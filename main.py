import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import filedialog

# Configuration
IMG_SIZE = 128
DATASET_PATH = r"D:\brain_tumor_dataset\dataset" 
MODEL_PATH = "model/brain_tumor_cnn.h5"
EPOCHS = 10
BATCH_SIZE = 16

# Load and preprocess dataset
def load_data():
    X, y = [], []
    for label, folder in enumerate(["no", "yes"]):  # 0 = no tumor, 1 = tumor
        folder_path = os.path.join(DATASET_PATH, folder)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = to_categorical(y, 2)
    return X, y

# Define CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Model evaluation: accuracy, confusion matrix, ROC curve
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Tumor", "Tumor"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Tumor", "Tumor"], yticklabels=["No Tumor", "Tumor"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

# Predict MRI image using the trained model
def predict_from_path(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Error: Cannot read the image. Please try again.")
        return
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    prediction = model.predict(img)
    result = np.argmax(prediction)
    print(f"\n🧠 Prediction Result: {'Tumorous' if result == 1 else 'Non-Tumorous'}")

# Open a file dialog to choose an image
def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select MRI Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    return file_path

# Main function
def main():
    print("[INFO] Loading dataset...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load or train model
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Training new model...")
        model = build_model()
        model.fit(X_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        print("[INFO] Model saved!")
    else:
        print("[INFO] Loading existing model...")
        model = load_model(MODEL_PATH)

    print("[INFO] Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Loop for image prediction
    while True:
        input("\n📂 Press Enter to choose an MRI image to test (or close the file dialog to exit)...")
        image_path = choose_file()
        if not image_path:
            print("👋 Exiting image test loop.")
            break
        predict_from_path(model, image_path)

if __name__ == "__main__":
    main()
