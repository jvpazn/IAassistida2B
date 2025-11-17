import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = datasets.mnist.load_data()

num_classes = 10
y_train = keras.utils.to_categorical(y_train_raw, num_classes)
y_test = keras.utils.to_categorical(y_test_raw, num_classes)

x_train = x_train_raw.reshape(60000, 784).astype("float32")/255
x_test = x_test_raw.reshape(10000, 784).astype("float32")/255

model_dnn = keras.Sequential([
    layers.Dense(512, activation='relu', input_dim=784),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_dnn.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(0.001),
    metrics=["accuracy"]
)

history = model_dnn.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
)

os.makedirs("model", exist_ok=True)
model_dnn.save("model/final_DNN_model.h5")

y_pred_prob = model_dnn.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nAcurácia:", accuracy_score(y_true, y_pred))
print("Precisão:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))
print("F1:", f1_score(y_true, y_pred, average='macro'))
print("Kappa:", cohen_kappa_score(y_true, y_pred))
print("\nRelatório completo:\n", classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusão - DNN")
plt.show()

y_true_bin = label_binarize(y_true, classes=range(num_classes))
fpr, tpr, roc_auc = {}, {}, {}

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Classe {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Falso Positivo")
plt.ylabel("Verdadeiro Positivo")
plt.title("Curva ROC - DNN")
plt.legend()
plt.show()

X_train = x_train_raw.reshape(60000, 28, 28, 1).astype("float32")/255
X_test = x_test_raw.reshape(10000, 28, 28, 1).astype("float32")/255

model_cnn = keras.Sequential([
    layers.Conv2D(32, 5, padding='same', activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, 3, padding='same', activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model_cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model_cnn.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(X_test, y_test)
)

model_cnn.save("model/final_CNN_model.h5")

def visualizar(n=20):
    y_pred_vis = np.argmax(model_cnn.predict(X_test[:n]), axis=-1)
    fig, ax = plt.subplots(nrows=n//5, ncols=5)
    ax = ax.flatten()

    print("\nResultados das previsões:")
    for i in range(n):
        print(y_pred_vis[i], end=", ")
        img = X_test[i].reshape(28, 28)
        ax[i].imshow(img, cmap='gray')
        ax[i].axis("off")

    plt.show()

visualizar(20)