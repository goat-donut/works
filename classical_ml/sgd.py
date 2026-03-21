import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.02 * np.exp(-x) - 0.2 * np.sin(3 * x) + 0.5 * np.cos(2 * x) - 7

def loss(w, x, y):
    return (x @ w - y)**2
# здесь объявляйте необходимые функции

def df(w, x, y):
    return 2 * (x @ w - y) * x.T
coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x)# значения функции по оси ординат
sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 1e-3, 1e-4, 1e-5, 1e-6]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
x = np.column_stack([np.ones(sz), coord_x, coord_x**2, coord_x**3, coord_x**4])
Qe = np.mean((x @ w - coord_y)**2)# начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел
for _ in range(N):
    k = np.random.randint(0, sz-1)
    xk = x[k]
    ax = (x @ w)[k]
    yk = coord_y[k]
    lossk = loss(w, xk, yk)
    w = w - eta * df (w, xk, yk)
    Qe = lm * lossk + (1 - lm) * Qe
Q = np.mean(loss(w, x, coord_y))
print(Q)