import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def generate_datasets():
    x_circles, y_circles = make_circles(n_samples=400, noise=0.1, factor=0.3, random_state=42)

    x_moons, y_moons = make_moons(n_samples=400, noise=0.1, random_state=42)

    x_linear, y_linear = make_classification(n_samples=400, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)

    datasets = {
        'Circles': (x_circles, y_circles),
        'Moons': (x_moons, y_moons),
        'Linear': (x_linear, y_linear)
    }

    return datasets

def plot_results(x, y, mlp_pred, rbf_pred, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Исходные данные
    axes[0].scatter(x[: ,0], x[: ,1], c=y, cmap='bwr', alpha=0.7)
    axes[0].set_title(f'{title} - Исходные данные')

    # MLP результаты
    axes[1].scatter(x[:, 0], x[:, 1], c=mlp_pred, cmap='bwr', alpha=0.7)
    axes[1].set_title('MLP - предсказания')

    # RBF результаты
    axes[2].scatter(x[:, 0], x[:, 1], c=rbf_pred, cmap='bwr', alpha=0.7)
    axes[2].set_title('RBF - предсказания')

    plt.tight_layout()
    plt.show()

def plot_decision_boundaries(x, y, mlp_model, rbf_model, title):
    # Сетка для визуализации разделяющих поверхностей
    h = 0.02
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (model, name) in enumerate(zip([mlp_model, rbf_model], ['MLP', 'RBF'])):
        z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        axes[i].contourf(xx, yy, z, alpha=0.3, cmap='bwr')
        axes[i].scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', alpha=0.7)
        axes[i].set_title(f'{name} - Разделяющая поверхность\n{title}')

    plt.tight_layout()
    plt.show()

def run_experiment(mlp_layers=(50,30), rbf_gamma=1.0):
    datasets = generate_datasets()

    # Создание моделей
    mlp = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=mlp_layers, activation='tanh', max_iter=1000, random_state=42, learning_rate_init=0.01))])

    # Для RBF используем SVM с RBF-ядром как аналог RBF-сети
    from sklearn.svm import SVC
    rbf = Pipeline([('scaler', StandardScaler()), ('rbf', SVC(kernel='rbf', gamma=rbf_gamma, C=1.0))])

    results = {}

    for name, (x, y) in datasets.items():
        print(f"\n{'='*50}")
        print(f"ДАТАСЕТ: {name}")
        print(f"{'='*50}")

        # Разделение на обучающую и тестовую выборки
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

        # Обучение MLP
        print("Обучение MLP...")
        mlp.fit(x_train, y_train)

        # Обучение RBF
        print("Обучение RBF...")
        rbf.fit(x_train, y_train)

        # Предсказания
        mlp_pred = mlp.predict(x_test)
        rbf_pred = rbf.predict(x_test)

        # Оценка качества
        print("\nMLP Результаты:")
        print(classification_report(y_test, mlp_pred))

        print("\nRBF-результаты:")
        print(classification_report(y_test, rbf_pred))

        # Визуализация
        plot_results(x_test, y_test, mlp_pred, rbf_pred, name)
        plot_decision_boundaries(x_test, y_test, mlp, rbf, name)

        # Сохранение результатов
        results[name] = {
            'mlp_score': mlp.score(x_test, y_test),
            'rbf_score': rbf.score(x_test, y_test),
            'mlp_predictions': mlp_pred,
            'rbf_predictions': rbf_pred
        }

    return results, mlp, rbf

def analyze_results(results):
    print("\n" + "="*60)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*60)

    metrics_data = []
    for dataset_name, result in results.items():
        metrics_data.append({
            'Dataset': dataset_name,
            'MLP_Accuracy': result['mlp_score'],
            'RBF_Accuracy': result['rbf_score']
        })

    # Таблица сравнения точности
    import pandas as pd
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.to_string(index=False))

    # Визуализация сравнения
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(results))
    width = 0.35

    mlp_scores = [results[name]['mlp_score'] for name in results.keys()]
    rbf_scores = [results[name]['rbf_score'] for name in results.keys()]

    ax.bar(x_pos - width/2, mlp_scores, width, label='MLP', alpha=0.7)
    ax.bar(x_pos + width/2, rbf_scores, width, label='RBF', alpha=0.7)

    ax.set_xlabel('Датасеты')
    ax.set_ylabel('Точность')
    ax.set_title('Сравнение точности MLP и RBF сетей')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results.keys())
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Запуск эксперимента
# results, mlp_model, rbf_model = run_experiment()
# analyze_results(results)

def study_architecture():
    # Разные конфигурации MLP
    architectures = [
        (10,),           # Один скрытый слой
        (50, 30),        # Два скрытых слоя  
        (100, 50, 25),   # Три скрытых слоя
    ]
    
    # Разные параметры гамма для RBF
    gammas = [0.1, 1.0, 10.0, 100.0]

    for architecture, gamma in zip(architectures, gammas):
        results, mlp_model, rbf_model = run_experiment(mlp_layers=architecture, rbf_gamma=gamma)
        print(f"АРХИТЕКТУРА MLP: {architecture} | ГАММА RBF: {gamma}")
        analyze_results(results)

    results, mlp_model, rbf_model = run_experiment(rbf_gamma=gammas[-1])
    print(f"ГАММА RBF: {gammas[-1]}")
    analyze_results(results)

study_architecture()