import numpy as np
import random
import matplotlib.pyplot as plt

# Параметры среды
GRID_SIZE = 5
ACTIONS = ['up', 'down', 'left', 'right'] # 0, 1, 2, 3
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 3), (2, 1)]

# Гиперпараметры Q-learning
EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Инициализация Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

def take_action(state, action):
    x, y = state

    if ACTIONS[action] == 'up':
        x -= 1
    elif ACTIONS[action] == 'down':
        x += 1
    elif ACTIONS[action] == 'left':
        y -= 1
    elif ACTIONS[action] == 'right':
        y += 1

    # Проверка выхода за границы
    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
        return state, -10, True  # штраф и конец эпизода

    next_state = (x, y)

    # Проверка на препятствие
    if next_state in OBSTACLES:
        return next_state, -10, True  # штраф и конец эпизода

    # Проверка на финиш
    if next_state == GOAL:
        return next_state, 100, True  # награда и конец эпизода

    # Обычное перемещение по клетке
    return next_state, -1, False

def choose_action(state, epsilon):
    # epsilon-жадная стратегия
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, len(ACTIONS) - 1)  # случайное действие
    else:
        return np.argmax(q_table[state[0], state[1]])  # лучшее действие

# Основной цикл обучения
def train_q_learning(alpha=ALPHA, gamma=GAMMA, epsilon_decay=EPSILON_DECAY):
    global EPSILON, q_table

    rewards = []
    EPSILON = 1.0
    q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))  # сброс таблицы

    for episode in range(1, EPISODES + 1):
        state = START
        total_reward = 0
        done = False

        while not done:
            action = choose_action(state, EPSILON)
            next_state, reward, done = take_action(state, action)

            # Обновление Q-table
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[next_state[0], next_state[1]])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state[0], state[1], action] = new_value

            state = next_state
            total_reward += reward

        # Сохраняем награду
        rewards.append(total_reward)

        # Затухание epsilon
        EPSILON = max(MIN_EPSILON, EPSILON * epsilon_decay)

        if episode % 250 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {EPSILON:.2f}")

    print("Обучение завершено!")
    return rewards

def print_grid(agent_pos):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) == agent_pos:
                print('@', end=' ')
            elif (i, j) == GOAL:
                print('F', end=' ')
            elif (i, j) in OBSTACLES:
                print('X', end=' ')
            else:
                print('.', end=' ')
        print()
    print()


def test_agent():
    state = START
    done = False
    steps = 0

    print("Тестирование обученного агента:\n")

    while not done and steps < 50:
        print_grid(state)

        # Всегда выбираем ЛУЧШЕЕ действие (epsilon = 0)
        action = np.argmax(q_table[state[0], state[1]])

        next_state, reward, done = take_action(state, action)

        state = next_state
        steps += 1

    print_grid(state)
    print(f"Тест завершён за {steps} шагов.")

rewards = train_q_learning(gamma=0.1)


# Кумулятивная сумма наград
cumulative_rewards = np.cumsum(rewards)

# Построение графика
plt.figure()
plt.plot(cumulative_rewards)
plt.xlabel("Номер эпизода")
plt.ylabel("Кумулятивная награда")
plt.title("Кумулятивная награда агента по эпизодам")
plt.grid(True)
plt.show()