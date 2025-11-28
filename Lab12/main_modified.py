import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

class FuzzyANDNeuron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(0.1, 0.5, input_size)
        self.x = None
        self.aggregated = None
        
    def forward(self, x):
        self.x = x
        # T-норма: произведение (product) по парам (x_i * w_i)
        self.aggregated = x * self.weights
        # S-норма: вероятностное ИЛИ по всем результатам: 1 - prod(1 - a_i)
        one_minus = 1.0 - self.aggregated
        # guard для численной стабильности
        one_minus = np.clip(one_minus, 1e-12, 1.0)
        output = 1.0 - np.prod(one_minus)
        return output
    
    def backward(self, grad_output, lr):
        # Для выходной функции out = 1 - prod(1 - a_i),
        # d out / d a_i = prod_{j != i} (1 - a_j)
        a = self.aggregated
        one_minus = 1.0 - a
        one_minus_clipped = np.clip(one_minus, 1e-12, 1.0)
        prod_all = np.prod(one_minus_clipped)
        # prod_excl_i = prod_all / (1 - a_i)  (безопасно через clipped)
        prod_excl = prod_all / one_minus_clipped
        grad_aggregated = grad_output * prod_excl  # dL/da_i
        
        # Градиент по весам: a_i = x_i * w_i => d a_i / d w_i = x_i
        grad_weights = grad_aggregated * self.x
        
        # Обновление весов
        self.weights -= lr * grad_weights
        self.weights = np.clip(self.weights, 0.01, 0.99)
        
        # Градиент по входу для передачи дальше: d a_i / d x_i = w_i
        grad_x = grad_aggregated * self.weights
        return grad_x


class FuzzyORNeuron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(0.5, 0.9, input_size)
        self.x = None
        self.aggregated = None
        
    def forward(self, x):
        self.x = x
        # S-норма по парам: вероятностное ИЛИ для каждой пары (x_i, w_i): x + w - x*w
        self.aggregated = self.x + self.weights - self.x * self.weights
        # T-норма по всем результатам: произведение (product) -> out = prod(a_i)
        a = self.aggregated
        # guard для численной стабильности
        a_clipped = np.clip(a, 1e-12, 1.0)
        output = np.prod(a_clipped)
        return output
    
    def backward(self, grad_output, lr):
        # Для out = prod(a_i), d out / d a_i = prod_{j != i} a_j
        a = self.aggregated
        a_clipped = np.clip(a, 1e-12, 1.0)
        prod_all = np.prod(a_clipped)
        prod_excl = prod_all / a_clipped  # prod over j != i
        grad_aggregated = grad_output * prod_excl  # dL/da_i
        
        # Для a_i = x_i + w_i - x_i*w_i => d a_i / d w_i = 1 - x_i
        grad_weights = grad_aggregated * (1.0 - self.x)
        
        # Обновление весов
        self.weights -= lr * grad_weights
        self.weights = np.clip(self.weights, 0.01, 0.99)
        
        # Градиент по входу для передачи назад: d a_i / d x_i = 1 - w_i
        grad_x = grad_aggregated * (1.0 - self.weights)
        return grad_x


class HybridFuzzyNetwork:
    def __init__(self, input_size, num_and_neurons):
        self.and_layer = [FuzzyANDNeuron(input_size) for _ in range(num_and_neurons)]
        self.or_neuron = FuzzyORNeuron(num_and_neurons)
        self.and_outputs = None
        
    def forward(self, x):
        self.and_outputs = np.array([neuron.forward(x) for neuron in self.and_layer])
        output = self.or_neuron.forward(self.and_outputs)
        return output
    
    def backward(self, grad_output, lr):
        # Backprop через OR-нейрон
        grad_and_outputs = self.or_neuron.backward(grad_output, lr)
        
        # Backprop через каждый AND-нейрон
        for i, neuron in enumerate(self.and_layer):
            neuron.backward(grad_and_outputs[i], lr)
    
    def train(self, X, y, epochs=20, lr=0.01):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for xi, yi in zip(X, y):
                # Forward
                output = self.forward(xi)
                
                # MSE loss
                loss = (output - yi) ** 2
                total_loss += loss
                
                # Градиент loss по output
                grad = 2 * (output - yi)
                
                # Backward
                self.backward(grad, lr)
            
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        return losses

# Генерация данных
X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                            n_redundant=0, n_classes=2, random_state=42)

# Нормализация в [0, 1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = y.astype(float)

# Создание и обучение сети
net = HybridFuzzyNetwork(input_size=4, num_and_neurons=5)
losses = net.train(X, y, epochs=50, lr=0.05)

# Оценка точности
predictions = np.array([net.forward(xi) for xi in X])
predictions_binary = (predictions > 0.5).astype(int)
accuracy = np.mean(predictions_binary == y)
print(f"\nAccuracy: {accuracy:.2%}")

print("\nТекущие веса AND-нейронов:")
for i, neuron in enumerate(net.and_layer):
    print(f"AND-нейрон {i+1}: {neuron.weights}")

print("\nТекущие веса OR-нейрона:")
print(net.or_neuron.weights)

# График функции потерь
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Реальная классификация
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=50)
axes[0].set_title("Реальная классификация (Признаки 1 и 2)")
axes[0].set_xlabel("Признак 1")
axes[0].set_ylabel("Признак 2")

# Предсказанная моделью классификация
axes[1].scatter(X[:, 0], X[:, 1], c=predictions_binary, cmap='bwr', edgecolor='k', s=50)
axes[1].set_title("Предсказание модели (Признаки 1 и 2)")
axes[1].set_xlabel("Признак 1")
axes[1].set_ylabel("Признак 2")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Реальная классификация для признаков 3 и 4
axes[0].scatter(X[:, 2], X[:, 3], c=y, cmap='bwr', edgecolor='k', s=50)
axes[0].set_title("Реальная классификация (Признаки 3 и 4)")
axes[0].set_xlabel("Признак 3")
axes[0].set_ylabel("Признак 4")

# Предсказанная моделью классификация для признаков 3 и 4
axes[1].scatter(X[:, 2], X[:, 3], c=predictions_binary, cmap='bwr', edgecolor='k', s=50)
axes[1].set_title("Предсказание модели (Признаки 3 и 4)")
axes[1].set_xlabel("Признак 3")
axes[1].set_ylabel("Признак 4")

plt.tight_layout()
plt.show()
