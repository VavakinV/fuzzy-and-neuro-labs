import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# ------------------------------------------
# 1. Генерация данных
# ------------------------------------------
# N = 10_000

# a = np.random.randint(1, 101, size=N)
# b = np.random.randint(1, 101, size=N)
# c = np.random.randint(1, 101, size=N)
# d = np.random.randint(1, 101, size=N)

# # target = 1 если a + c > b + d, иначе 0
# y = (a + c > b + d).astype(int)

# X = np.column_stack((a, b, c, d))

# # Разделение на обучающую и тестовую выборки (80/20)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Преобразование в формат для RNN: (samples, timesteps=4, features=1)
# X_train = X_train.reshape((X_train.shape[0], 4, 1))
# X_test = X_test.reshape((X_test.shape[0], 4, 1))

# ------------------------------------------
# 2. Построение и обучение простой RNN
# ------------------------------------------
# model = Sequential()
# model.add(SimpleRNN(32, return_sequences=False, input_shape=(4, 1)))
# model.add(Dense(1, activation="sigmoid"))

# model.compile(optimizer="adam",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])

# history = model.fit(
#     X_train,
#     y_train,
#     validation_split=0.2,
#     epochs=15,
#     batch_size=32
# )

# # График ошибки
# plt.figure()
# plt.plot(history.history["loss"], label="train loss")
# plt.plot(history.history["val_loss"], label="val loss")
# plt.title("Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# # График точности
# plt.figure()
# plt.plot(history.history["accuracy"], label="train accuracy")
# plt.plot(history.history["val_accuracy"], label="val accuracy")
# plt.title("Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# loss, acc = model.evaluate(X_test, y_test)
# print("Test accuracy:", acc)

# ------------------------------------------
# 3. Многослойная RNN и возврат последовательностей
# ------------------------------------------
# model = Sequential()

# model.add(SimpleRNN(32, return_sequences=True, input_shape=(4, 1)))
# model.add(SimpleRNN(32, return_sequences=False))

# model.add(Dense(1, activation="sigmoid"))

# model.compile(optimizer="adam",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])

# history = model.fit(
#     X_train,
#     y_train,
#     validation_split=0.2,
#     epochs=15,
#     batch_size=32
# )

# loss, acc = model.evaluate(X_test, y_test)
# print("Test accuracy:", acc)

# plt.figure()
# plt.plot(history.history["loss"], label="train loss")
# plt.plot(history.history["val_loss"], label="val loss")
# plt.title("Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(history.history["accuracy"], label="train accuracy")
# plt.plot(history.history["val_accuracy"], label="val accuracy")
# plt.title("Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# ------------------------------------------
# 4. Предсказание временного ряда (Регрессия)
# ------------------------------------------
t = np.linspace(0, 100, 2000)
series = np.sin(t) + 0.1 * np.random.randn(len(t))

n = 20

X = []
y = []

for i in range(len(series) - n):
    X.append(series[i:i+n])
    y.append(series[i+n])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential()
model.add(SimpleRNN(32, return_sequences=True, input_shape=(n, 1)))
model.add(SimpleRNN(32))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

predictions = model.predict(X_test)

plt.figure()
plt.plot(y_test, label="Real")
plt.plot(predictions, label="Predicted")
plt.title("RNN Sinusoid Regression")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.legend()
plt.show()