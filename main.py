import numpy as np
import scipy as sp

from scipy import integrate
from scipy.optimize import minimize_scalar

np.warnings.filterwarnings("ignore")

# List of mechanisms
G = [[lambda a: a ** (1 / 4), "a^(1/4)"],
     [lambda a: a ** (1 / 3), "a^(1/3)]"],
     [lambda a: a ** (1 / 2), "a^(1/2)"],
     [lambda a: a, "a"],
     [lambda a: a ** (3 / 2), "a^(3/2)"],
     [lambda a: a ** 2, "a^2"],
     [lambda a: 1 - (1 - a) ** (1 / 3), "1-(1-a)^(1/3)"],
     [lambda a: 1 - (1 - a) ** (1 / 2), "1-(1-a)^(1/2)"],
     [lambda a: 1 - (1 - a) ** (1 / 3), "1-(1-a)^(1/3)"],
     [lambda a: -np.log((1 - a) ** (1 / 4)), "-ln( (1-a)^(1/4) )"],
     [lambda a: -np.log((1 - a) ** (1 / 3)), "-ln( (1-a)^(1/3) )"],
     [lambda a: -np.log((1 - a) ** (1 / 2)), "-ln( (1-a)^(1/2) )"],
     [lambda a: -np.log((1 - a) ** (2 / 3)), "-ln( (1-a)^(2/3) )"],
     [lambda a: -np.log(1 - a), "-ln(1-a)"],
     [lambda a: (1 - a) * np.log(1 - a) + a, "(1-a)*ln(1-a) + a"],
     [lambda a: 1 - 2 / 3 * a - (1 - a) ** (2 / 3), "1 - 2/3*a - (1-a)^(2/3)"],
     [lambda a: (1 - (1 - a) ** (1 / 3)) ** 2, "(1-(1-a)^(1/3))^2"],
     [lambda a: ((1 + a) ** (1 / 3) - 1) ** 2, "((1+a)^(1/3) - 1)^2"],
     [lambda a: 1 - (1 - a) ** 2, "1-(1-a)^2"],
     [lambda a: 1 - (1 - a) ** 3, "1-(1-a)^3"],
     [lambda a: 1 - (1 - a) ** 4, "1-(1-a)^4"],
     ]

R = 8.3144598


def calc_x_vec(temp_vec, act_energy):
    """
    Calculates U(T,E)
    :param temp_vec: np.array of temperatures
    :param act_energy: Energy of activation
    :return: np.array of x values
    """
    return act_energy / (R * temp_vec)


def integrand(u):
    return sp.exp(-u) / u


def calc_p(temp_vec, act_energy):
    """
    Calculates p(x) value
    :param temp_vec: np.array of temperature
    :param act_energy:  activation energy
    :return: np.array of p(x) or p(T, Eact) values
    """
    x = calc_x_vec(temp_vec, act_energy)
    p = []

    for i in x:
        p.append((sp.exp(-i) / i) - integrate.quad(integrand, i, sp.inf)[0])
    return p


def calc_f(algorithm_number, alpha_vec):
    """
    Calculate G(alpha)
    :param algorithm_number: number of alg from G list
    :param alpha_vec: np_array of alpha values
    :return: returns array of G(alpha)
    """
    return G[algorithm_number][0](alpha_vec)


def calc_b(f_vec, p_vec):
    """
    Calculates B vector
    :param f_vec: vector of f(alpha)
    :param p_vec: vector of p(x)
    :return: B_vec
    """
    return np.log10(f_vec) - np.log10(p_vec)


def calc_delta(act_energy, t_vec, f_vec):
    """
    Calculates delta value
    :param act_energy: Activation energy
    :param t_vec: vec of temperatures
    :param f_vec: vec of f(alpha)
    :return: delta value
    """
    p_vec = calc_p(t_vec, act_energy)
    B = calc_b(f_vec, p_vec)
    return np.std(B)


def g_as_str(algorithm_number):
    """
    Prints G(alpha) as string
    :param algorithm_number: alg number
    :return: string
    """
    return G[algorithm_number][1]


def extract_data(filename):
    """
    Extract two arrays from .csv: T and alpha

    :param filename: name of .csv file
    :return: np.array, 1st col - T, 2nd col - alpha
    """

    data = np.genfromtxt(filename, delimiter=";", skip_header=1)
    return data


content = extract_data("UN-1a.csv")
T = content[:, 0]
a = content[:, 1]

for alg_n in range(len(G)):
    print("Work with ", g_as_str(alg_n))
    F = calc_f(alg_n, a)
    #opt = minimize_scalar(calc_delta, args=(T, F,),
    #                      method="brent", bracket=(160,220))
    print(calc_delta(166000, T, F))
    #print(opt.x, opt.fun)

