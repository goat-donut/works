from sympy import Symbol, solve, pi, exp

# Функция p(x)
def p_x(x, D_x, x_0):
    return (1 / (2 * pi * D_x)**.5) * exp(-((x - x_0)**2) / (2 * D_x))

# Функция p(y1 | x)
def p_y1_given_x(x, y1, D_e):
    return (1 / (2 * pi * D_e)**.5) * exp(-((x - y1)**2) / (2 * D_e))

# Функция для вычисления производной логарифма произведения
def hat_x_derivative(x, y1, D_x, D_e, x_0):
    # Вычисляем p(x) и p(y1 | x)
    p_x_value = p_x(x, D_x, x_0)
    p_y1_given_x_value = p_y1_given_x(x, y1, D_e)

    # Вычисляем производные
    dp_x_dx = (-1 / ((2 * pi * D_x)**.5)) * exp(-((x - x_0)**2) / (2 * D_x)) * (x - x_0) / D_x
    dp_y1_given_x_dx = (-1 / ((2 * pi * D_e)**.5)) * exp(-((x - y1) ** 2) / (2 * D_e)) * (x - y1) / D_e

    # Используем правило суммы для производной логарифма произведения
    return (dp_y1_given_x_dx / p_y1_given_x_value) + (dp_x_dx / p_x_value)

D_e = 0.2 # необъяснённая дисперсия (шум)
D_x = 0.5 # объяснённая дисперсия (предиктором x)
x_0 = 7   # среднее (матожидание)
y1 = 6.5  # наблюдаемое значение сигнала от ГЛОНАСС

x = Symbol('x')
solve(hat_x_derivative(x, y1, D_x, D_e, x_0))
print(solve(hat_x_derivative(x, y1, D_x, D_e, x_0)))
