import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk interpolasi Lagrange
def lagrange_interpolation(x, y, xp):
    yp = 0
    for i in range(len(x)):
        p = 1
        for j in range(len(x)):
            if i != j:
                p *= (xp - x[j]) / (x[i] - x[j])
        yp += y[i] * p
    return yp

# Fungsi untuk membagi beda yang digunakan dalam interpolasi Newton
def divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y

    for j in range(1, n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    
    return coef[0, :]

# Fungsi untuk interpolasi Newton
def newton_interpolation(x, y, xp):
    coef = divided_diff(x, y)
    n = len(x)
    yp = coef[0]
    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (xp - x[j])
        yp += term
    return yp

# Data yang diberikan
x_data = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y_data = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Rentang x untuk plot
x_plot = np.linspace(5, 40, 100)
y_lagrange = [lagrange_interpolation(x_data, y_data, xi) for xi in x_plot]
y_newton = [newton_interpolation(x_data, y_data, xi) for xi in x_plot]

# Plot hasil interpolasi
plt.figure(figsize=(14, 7))
plt.plot(x_plot, y_lagrange, label='Interpolasi Lagrange', color='blue', linestyle='-')
plt.plot(x_plot, y_newton, label='Interpolasi Newton', color='red', linestyle='--')
plt.scatter(x_data, y_data, color='black', label='Data Points')
plt.xlabel('Tegangan (kg/mm^2)')
plt.ylabel('Waktu Patah (jam)')
plt.title('Interpolasi Polinom Lagrange dan Newton')
plt.legend()
plt.grid(True)
plt.show()
