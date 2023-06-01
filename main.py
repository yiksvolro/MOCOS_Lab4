import numpy as np
import matplotlib.pyplot as plt
from statistics import median

def generate_sequence(n, A, phi, x_interval):
    N = n * 100
    x = np.linspace(x_interval[0], x_interval[1], N)
    exact_values = A * np.sin(n * x + phi)
    errors = np.random.uniform(-0.05 * A, 0.05 * A, N)
    noisy_values = exact_values + errors
    return x, noisy_values

# Фунція знаходження середнього арифметичного
def compute_arithmetic_mean(sequence):
    return np.mean(sequence)

# Функція знаходження середнього гармонійного
def compute_harmonic_mean(sequence):
    return len(sequence) / np.sum(1 / sequence)

# Функція знаходження середнього геометричного
def compute_geometric_mean(sequence):
    valid_values = sequence[np.logical_and(sequence > 0, ~np.isnan(sequence))]
    return np.exp(np.mean(np.log(valid_values)))


# Функція побудови графіку
def plot_graph(x, exact_values, noisy_values):
    plt.plot(x, noisy_values, label='Наближене', color='r')
    plt.plot(x, exact_values, label='Точне', color='g')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Функція обчислення точного значення
def compute_exact_values(x, A, n, phi):
    return A * np.sin(n * x + phi)

# Функція порівняння точного і наближеного значень
def compare_values(approx_values, exact_values, epsilon=1e-10):
    absolute_errors = np.abs(exact_values - approx_values)
    relative_errors = absolute_errors / (np.abs(exact_values) + epsilon)
    return absolute_errors, relative_errors


# Параметри генерації послідовності
n = 16
A = 4.0
phi = 7
x_interval = (0, 0.75)

# Генерація послідовності
x, y_noisy = generate_sequence(n, A, phi, x_interval)

# Обчислення середніх значень
arithmetic_mean = compute_arithmetic_mean(y_noisy)
print("Середнє арифметичне значення: ",arithmetic_mean)
harmonic_mean = compute_harmonic_mean(y_noisy)
print("Середнє гармонічне значення: ",harmonic_mean)
geometric_mean = compute_geometric_mean(y_noisy)
print("Середнє геометричне значення: ",geometric_mean)

# Обчислення точних значень та порівняння з наближеними значеннями
exact_values = compute_exact_values(x, A, n, phi)
print("Середнє точне значення: ",abs(median(exact_values)))
absolute_errors, relative_errors = compare_values(y_noisy, exact_values)
print("Середня абсолютна похибка: ",median(absolute_errors))
print("Середня відносна похибка: ",median(relative_errors))


# Порівняння максимумів та мінімумів похибок
max_absolute_error = np.max(absolute_errors)
min_absolute_error = np.min(absolute_errors)
max_relative_error = np.max(relative_errors)
min_relative_error = np.min(relative_errors)
print("Максимальне значення абсолютної похибки: ",max_absolute_error)
print("Мінімальне значення абсолютної похибки: ",min_absolute_error)
print("Максимальне значення відносної похибки: ",max_relative_error)
print("Мінімальне значення відносної похибки:",min_relative_error)

# Візуалізація результатів
plot_graph(x, exact_values, y_noisy)