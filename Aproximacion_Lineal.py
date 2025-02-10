import numpy as np
from scipy.integrate import quad
import itertools
from sympy import symbols, Matrix, solve, sin, cos
import matplotlib.pyplot as plt

# Definir la variable simbólica
x = symbols('x')

# Definir las funciones en términos de SymPy
def f1(x):
    return 1

def f2(x):
    return sin(x)

def f3(x):
    return cos(x)

def f4(x):
    return sin(2*x)

def f5(x):
    return cos(2*x)

def f6(x):
    return x**3

# Crear el array de funciones
functions = [f1, f2, f3, f4, f5]

# Definir los límites de integración
a = -np.pi
b = np.pi

# Función para integrar el producto de dos funciones en el intervalo [-pi, pi]
def integrate_product(f, g):
    product = lambda x: f(x) * g(x)
    integral, _ = quad(product, a, b)

    # Si la integral es menor que 10^-10, se devuelve 0
    if abs(integral) < 1e-10:
        integral = 0
    return integral

# Generar todos los pares posibles de funciones
pairs = itertools.combinations(functions, 2)

# Variable para comprobar si todas las integrales son cero
all_orthogonal = True

# Integrar el producto de cada par de funciones y mostrar el resultado
for f, g in pairs:
    integral = integrate_product(f, g)
    print(f'Integral de {f.__name__} * {g.__name__} entre -pi y pi: {integral}')
    print('---')

    # Si alguna integral no es cero, marcar que no todas las funciones son ortogonales
    if integral != 0:
        all_orthogonal = False

# Verificar si todas las funciones son ortogonales
if all_orthogonal:
    print("¡Todas las funciones son ortogonales entre sí!")
else:
    print("No todas las funciones son ortogonales entre sí.")

# Parte 2
# Definir la variable simbólica
x = symbols('x')

# Definir las funciones base
functions = [
    lambda x: 1,
    lambda x: np.sin(x),
    lambda x: np.cos(x),
    lambda x: np.sin(2*x),
    lambda x: np.cos(2*x)
]

# Definir la función objetivo
target_function = lambda x: x**3


# Función para calcular el producto interno de dos funciones en [-pi, pi]
def inner_product(f, g):
    integral, _ = quad(lambda x: f(x) * g(x), a, b)
    return integral

# Construir la matriz de Gram G (productos escalares <fi, fj>)
G = np.array([[inner_product(fi, fj) for fj in functions] for fi in functions])

# Construir el vector b (productos escalares <x^3, fi>)
b = np.array([inner_product(target_function, fi) for fi in functions])

# Resolver el sistema G * a = b
coefficients = np.linalg.solve(G, b)

# Mostrar los coeficientes calculados
print("Coeficientes obtenidos:")
for i, coef in enumerate(coefficients, 1):
    print(f"a{i} = {coef:.6f}")

# Función aproximada con la combinación lineal de las funciones base
def approximated_function(x):
    return sum(c * f(x) for c, f in zip(coefficients, functions))

# Valores de x para graficar
x_values = np.linspace(-np.pi, np.pi, 400)
y_target = target_function(x_values)
y_approx = [approximated_function(x) for x in x_values]

# Hacer la gráfica más cute 🌸
plt.figure(figsize=(8, 5), facecolor="#FFF5FD")  # Fondo rosado claro
plt.plot(x_values, y_target, label="$x^3$", linestyle="dashed", color="#FF69B4", linewidth=2)  # Rojo rosado
plt.plot(x_values, y_approx, label="Aproximación", color="#6A5ACD", linewidth=2)  # Azul lila

# Personalización cute ✨
plt.xlabel("x", fontsize=12, color="#FF1493")
plt.ylabel("y", fontsize=12, color="#FF1493")
plt.title("Aproximación de $x^3$\nFunciones base ortogonales", fontsize=14, color="#FF69B4")
plt.legend(fontsize=10, facecolor="white", edgecolor="pink")
plt.grid(color="#FFC0CB", linestyle="dotted")

# Cambiar fondo de la gráfica
plt.gca().set_facecolor("#FFF0F5")  # Lavanda claro

# Mostrar la gráfica
plt.show()

