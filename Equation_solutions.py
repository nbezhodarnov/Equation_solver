import numpy as np
import matplotlib.pyplot as plt
import math

#tg(x) + x ** 2 - 1

def function(x):
	return np.tan(x) + x ** 2 - 1

def canonical_function(x):
	return np.arctan(1 - x ** 2)

def function_derivative(x):
	return 1 / (np.cos(x) ** 2) + 2 * x

def function_2_derivative(x):
	return 2 - 2 * np.sin(2 * x) / (np.cos(x) ** 4)

def Newton_method(x, error):
	root = np.array([x])
	root = np.append(root, x - function(x) / function_derivative(x))
	i = 1
	while (abs(root[i] - root[i - 1]) > error):
		root = np.append(root, root[i] - function(root[i]) / function_derivative(root[i]))
		i += 1
	print("Newton's method:\n", "	Root: ", root[i], "\n	Number of steps: ", i)
	return root

def Newton_method_with_derivative_const(x, error):
	root = np.array([x])
	x_derivative = function_derivative(x)
	root = np.append(root, x - function(x) / x_derivative)
	i = 1
	while (abs(root[i] - root[i - 1]) > error):
		root = np.append(root, root[i] - function(root[i]) / x_derivative)
		i += 1
	print("Newton's method with derivative constant:\n", "	Root: ", root[i], "\n	Number of steps: ", i)
	return root

def Newton_method_with_secants(x, error):
	root = np.array([x])
	f_n_minus_1 = function(x)
	root = np.append(root, x - f_n_minus_1 / function_derivative(x))
	f_n = function(root[1])
	root = np.append(root, (root[0] * f_n - root[1] * f_n_minus_1) / (f_n - f_n_minus_1))
	i = 2
	while (abs(root[i] - root[i - 1]) > error):
		f_n_minus_1 = f_n
		f_n = function(root[i])
		root = np.append(root, (root[i - 1] * f_n - root[i] * f_n_minus_1) / (f_n - f_n_minus_1))
		i += 1
	print("Newton's method with secants:\n", "	Root: ", root[i], "\n	Number of steps: ", i - 1)
	return root

def Simple_iterations_method(x, error):
	root = np.array([x])
	root = np.append(root, canonical_function(x))
	i = 1
	while (abs(root[i] - root[i - 1]) > error):
		root = np.append(root, canonical_function(root[i]))
		i += 1
	print("Simple iterations method:\n", "	Root: ", root[i], "\n	Number of steps: ", i)
	return root

def Simple_iterations_method_with_secants(x, error):
	root = np.array([x])
	root = np.append(root, canonical_function(x))
	root = np.append(root, (x * canonical_function(root[1]) - root[1] ** 2) / (canonical_function(root[1]) - 2 * root[1] + x))
	i = 2
	while (abs(root[i] - root[i - 1]) > error):
		fi_n_minus_1 = canonical_function(root[i - 1])
		fi_n = canonical_function(root[i])
		root = np.append(root, (root[i - 1] * fi_n - root[i] * fi_n_minus_1) / (fi_n - root[i] - fi_n_minus_1 + root[i - 1]))
		i += 1
	print("Simple iterations method with secants:\n", "	Root: ", root[i], "\n	Number of steps: ", i - 1)
	return root

def Steffenson_method(x, error):
	root = np.array([x])
	fi_n = canonical_function(root[0])
	fi_fi_n = canonical_function(fi_n)
	root = np.append(root, (x * fi_fi_n - fi_n ** 2) / (fi_fi_n - 2 * fi_n + x))
	i = 1
	while (abs(root[i] - root[i - 1]) > error):
		fi_n = canonical_function(root[i])
		fi_fi_n = canonical_function(fi_n)
		root = np.append(root, (root[i] * fi_fi_n - fi_n ** 2) / (fi_fi_n - 2 * fi_n + root[i]))
		i += 1
	print("Steffenson method:\n", "	Root: ", root[i], "\n	Number of steps: ", i)
	return root
	
def main():
	x0 = 0.6
	error = 0.5e-4
		
	x_plot_table = np.linspace(0, 1, 50, dtype = float)
	y_plot_table = np.linspace(0, 0, 50, dtype = float)
	y_plot_table_of_part_1 = np.linspace(0, 0, 50, dtype = float)
	y_plot_table_of_part_2 = np.linspace(0, 0, 50, dtype = float)
	y_plot_table_canonical = np.linspace(0, 0, 50, dtype = float)
	y_plot_table_canonical_d = np.linspace(0, 0, 50, dtype = float)
	y_plot_table_canonical_2 = np.linspace(0, 0, 50, dtype = float)
	y_plot_table_canonical_2_d = np.linspace(0, 0, 50, dtype = float)
	y_plot_table_d = np.linspace(0, 0, 50, dtype = float)
	y_plot_table_d_2 = np.linspace(0, 0, 50, dtype = float)
	for i in range(50):
		y_plot_table[i] = function(x_plot_table[i])
		y_plot_table_of_part_1[i] = np.tan(x_plot_table[i])
		y_plot_table_of_part_2[i] = -x_plot_table[i] ** 2 + 1
		y_plot_table_canonical[i] = canonical_function(x_plot_table[i])
		y_plot_table_canonical_d[i] = -2 * x_plot_table[i] / (x_plot_table[i] ** 4 - 2 * x_plot_table[i] ** 2 + 2)
		y_plot_table_canonical_2[i] = np.sqrt(1 - np.tan(x_plot_table[i]))
		y_plot_table_canonical_2_d[i] = -1 / (np.cos(x_plot_table[i]) ** 2 * np.sqrt(1 - np.tan(x_plot_table[i])))
		y_plot_table_d[i] = function_derivative(x_plot_table[i])
		y_plot_table_d_2[i] = function_2_derivative(x_plot_table[i])
	
	fig, ax = plt.subplots()
	
	plt.figure(1)
	plt.plot(x_plot_table, y_plot_table_of_part_1, 'b-', label = 'tg(x)')
	plt.plot(x_plot_table, y_plot_table_of_part_2, 'm-', label = '-x ^ 2 + 1')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(loc='upper right')
	#plt.show()
	
	plt.figure(2)
	plt.plot(x_plot_table, y_plot_table, 'b-', label = 'tg(x) + x ^ 2 - 1')
	
	x = Newton_method(x0, error)
	plt.plot(x, np.linspace(2, 2, x.size, dtype = float), 'm*', label = 'N')
	x = Newton_method_with_secants(x0, error)
	plt.plot(x, np.linspace(1.7, 1.7, x.size, dtype = float), 'g*', label = 'Ns')
	x = Newton_method_with_derivative_const(x0, error)
	plt.plot(x, np.linspace(1.4, 1.4, x.size, dtype = float), 'y*', label = 'Ndc')
	x = Simple_iterations_method(x0, error)
	plt.plot(x, np.linspace(1.1, 1.1, x.size, dtype = float), 'r*', label = 'S')
	x = Simple_iterations_method_with_secants(x0, error)
	plt.plot(x, np.linspace(0.8, 0.8, x.size, dtype = float), 'c*', label = 'Ss')
	x = Steffenson_method(x0, error)
	plt.plot(x, np.linspace(0.5, 0.5, x.size, dtype = float), 'k*', label = 'Sf')
	
	plt.plot(x_plot_table, np.linspace(0, 0, 50, dtype = float), 'k-')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(loc='upper right')

	plt.figure(3)
	plt.plot(x_plot_table, y_plot_table_canonical, 'b-', label = 'arctan(1 - x ^ 2)')
	plt.plot(x_plot_table, y_plot_table_canonical_d, 'r-', label = '-2 * x / (x ^ 4 - 2 * x ^ 2 + 2)')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(loc='upper right')

	plt.figure(4)
	plt.plot(x_plot_table, y_plot_table, 'b-', label = 'f(x)')
	plt.plot(x_plot_table, y_plot_table_d, 'r-', label = 'f|(x)')
	plt.plot(x_plot_table, y_plot_table_d_2, 'm-', label = 'f"(x)')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(loc='upper right')

	plt.figure(5)
	plt.plot(x_plot_table, y_plot_table_canonical_2, 'b-', label = 'sqrt(1 - tg(x))')
	plt.plot(x_plot_table, y_plot_table_canonical_2_d, 'r-', label = '-1 / (cos(x) * sqrt(1 - tg(x)))')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(loc='upper right')

	plt.show()
	
if __name__ == '__main__':
    main()
