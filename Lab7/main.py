import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)

# Подготовка датасета
def target_function(x):
    return 2**x * torch.sin(2**-x)

with torch.no_grad():
    x_train = torch.linspace(-10, 10, 100)
    y_train = target_function(x_train)

noise = torch.randn(y_train.shape) / 5

# Построение графиков начальной функции и шума
# plt.plot(x_train.numpy(), y_train.numpy(), '-')

# plt.plot(x_train.numpy(), noise.numpy(), 'o')
# plt.axis([-10, 10, -1, 1])

y_train = y_train + noise

# plt.plot(x_train.numpy(), y_train.numpy(), '-')
# plt.legend(['Original function', 'Gaussian noise', 'Noise function'])

# plt.xlabel('x_train')
# plt.ylabel('y_train')
# plt.show()

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 10, 100).unsqueeze_(1)
y_validation = target_function(x_validation)

# Функция метрики
def metric(pred, target):
    return (pred - target).abs().mean()

# Класс сети
class Net(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons*2)
        self.act2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neurons*2, n_hidden_neurons)
        self.act3 = torch.nn.Tanh()
        self.fc4 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x
    
net = Net(100)

# Функция предсказания
def predict(net, x, y):
    with torch.no_grad():
        y_pred = net(x)

    x_np = x.detach().squeeze().numpy()
    y_np = y.detach().squeeze().numpy()
    y_pred_np = y_pred.detach().squeeze().numpy()

    plt.plot(x_np, y_np, 'o', label='Truth')
    plt.plot(x_np, y_pred_np, 'o', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()

# Предсказание без обучения
# predict(net, x_validation, y_validation)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0015)

# Функция потерь
def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()

# Цикл обучения
for epoch_index in range(2000):
    optimizer.zero_grad()

    y_pred = net.forward(x_train)
    loss_val = loss(y_pred, y_train)

    loss_val.backward()

    optimizer.step()

predict(net, x_validation, y_validation)
print(f"MAE: {metric(net.forward(x_validation), y_validation).item()}")