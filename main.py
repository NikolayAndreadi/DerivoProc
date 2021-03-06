import logging
import os
import sys

import numpy as np
import scipy as sp
from scipy import integrate
from scipy.optimize import minimize_scalar
from scipy.stats import linregress
import matplotlib.pyplot as plt

np.warnings.filterwarnings("ignore")
logging.basicConfig(filename="derivoproc.out", level=logging.INFO, format='', filemode='w')

TRUNC_VAL = 0.1
SUGGEST_COUNT = 3
R = 8.3144598

# List of mechanisms
G = [[lambda a: a ** (1 / 4), "a^(1/4) [P4]", lambda a: a ** 4],
     [lambda a: a ** (1 / 3), "a^(1/3)] [P3]", lambda a: a ** 3],
     [lambda a: a ** (1 / 2), "a^(1/2) [P2]", lambda a: a ** 2],
     [lambda a: a, "a [F0/R1]", lambda a: a],
     [lambda a: a ** 2, "a^2 [D1]", lambda a: a ** (1 / 2)],
     [lambda a: 1 - (1 - a) ** (1 / 3), "1-(1-a)^(1/3) [R3]", lambda a: a ** 3 - 3 * (a ** 2) + 3 * a],
     [lambda a: 1 - (1 - a) ** (1 / 2), "1-(1-a)^(1/2) [R2]", lambda a: 2*a - (a**2)],
     [lambda a: -np.log((1 - a) ** (1 / 4)), "-ln( (1-a)^(1/4) ) [A4]", lambda a: np.exp(-4 * a) * (np.exp(4 * a) - 1)],
     [lambda a: -np.log((1 - a) ** (1 / 3)), "-ln( (1-a)^(1/3) ) [A3]", lambda a: np.exp(-3 * a) * (np.exp(3 * a) - 1)],
     [lambda a: -np.log((1 - a) ** (1 / 2)), "-ln( (1-a)^(1/2) ) [A2]", lambda a: np.exp(-2 * a) * (np.exp(2 * a) - 1)],
     [lambda a: -np.log(1 - a), "-ln(1-a) [F1]", lambda a: np.exp(-a) * (np.exp(a) - 1)],
     [lambda a: (1 - a) * np.log(1 - a) + a, "(1-a)*ln(1-a) + a [D2]", lambda a: "N/A"],
     [lambda a: 1 - 2 / 3 * a - (1 - a) ** (2 / 3), "1 - 2/3*a - (1-a)^(2/3) [D4]", lambda a: "N/A"],
     [lambda a: (1 - (1 - a) ** (1 / 3)) ** 2, "(1-(1-a)^(1/3))^2 [D3]", lambda a: -3*a+(a**3 + 6*(a**2)+9*a)**(1/2)],
     [lambda a: 1 / (1 - a) - 1, "1/(1-a)-1 [F2]", lambda a: a/(a+1)],
     [lambda a: (1 / 2)*(1 - a)**(-2) - 1, "1/2(1-a)^-2 -1 [F3]", lambda a: 1 + 1/(np.sqrt(2) * np.sqrt(a+1))],
     [lambda a: (1 / 2)*(1 - a)**(-2) - 1, "1/2(1-a)^-2 -1 [F3]", lambda a: 1 + 1/(np.sqrt(2) * np.sqrt(a+1))],
     ]


def g_as_str(algorithm_number):
    """
    Prints G(alpha) as string
    :param algorithm_number: alg number
    :return: string
    """
    return G[algorithm_number][1]


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
    :return: np.array of p(x) or p(T, E_act) values
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


B = 0.0
P = np.array([], dtype=float)


def calc_delta(act_energy, t_vec, f_vec, save_g=False):
    """
    Calculates delta value
    :param act_energy: Activation energy
    :param t_vec: vec of temperatures
    :param f_vec: vec of f(alpha)
    :param save_g: if true - save Bmean * p(x)
    :return: delta value
    """
    p_vec = calc_p(t_vec, act_energy)
    b = calc_b(f_vec, p_vec)
    if save_g:
        global B, P
        B = np.average(b)
        P = p_vec
    return np.std(b)


def calc_init_energy(t_vec, alpha_vec):
    """
    Estimates initial activation energy
    :param t_vec: temperature vector
    :param alpha_vec: alpha vector
    :return: value of E_act
    """
    logging.info("Calculation initial Eact...")
    x = 1 / t_vec
    y = np.log(alpha_vec / (t_vec ** 2))

    dy = np.diff(y) / np.diff(x)
    target = (np.abs(dy - np.average(dy))).argmin()
    logging.info("  Using %d of %d points for calculating", target, len(t_vec))
    regress = linregress(x[:target], y[:target])
    e_init = regress[0] * (-R)
    logging.info("  Initial energy is %.2f J, R-squared is %.4f", e_init, regress[2] ** 2)
    return e_init


def find_best_alg(act_energy, temp_array, alpha_array):
    """
    Finds several algorithms with best delta
    :param act_energy: activation energy
    :param temp_array: temperature array
    :param alpha_array: alpha array
    :return: [algorithm_numbers]
    """
    logging.info("Finding best algorithms...")
    logging.info("  ----------------")
    summary = np.array([])
    for alg_n in range(len(G)):
        logging.info("  Trying: %s", g_as_str(alg_n))
        f = calc_f(alg_n, alpha_array)
        delta = calc_delta(act_energy, temp_array, f)
        logging.info("  Delta is %.5f", delta)
        logging.info("  ----------------")
        summary = np.append(summary, delta)
    best = np.argsort(summary)[:SUGGEST_COUNT]
    logging.info("Best algorithms are: %s", ', '.join([g_as_str(x) for x in best]))
    return best


def optimize_energy(temp_arr, f_arr, alg, fn):
    """
    Final optimization of activation energy
    :param temp_arr: temperature array
    :param f_arr: f-array
    :param alg: algorithm for reversing g(alpha)
    :param fn: filename without ending
    :return: [optimized energy, delta]
    """
    opt = minimize_scalar(calc_delta, args=(temp_arr, f_arr, True,))
    logging.info("  Done in %d itetations", opt.nit)
    logging.info("  Delta is %.5f", opt.fun)
    logging.info("  Eact is %.2f J", opt.x)

    logging.info("  Saving theoretical data to file...")
    global B, P
    P = [x*sp.power(10, B) for x in P]
    P = np.array(P)
    alpha = G[alg][2](P)
    name = "./" + fn + "__" + str(alg) + ".xy"
    f = open(name, "w+")
    np.savetxt(f, np.stack((temp_arr, alpha), axis=-1), delimiter=";")
    plt.plot(temp_arr, alpha, label="theory")
    f.close()
    return [opt.x, opt.fun]


def process_file(filename):
    logging.info("Reading content...")
    content = np.genfromtxt(filename, delimiter=";", skip_header=1)
    t = content[:, 0]
    a = content[:, 1]

    a = a[a > TRUNC_VAL]
    t = t[-len(a):]

    energy = calc_init_energy(t, a)
    best = find_best_alg(energy, t, a)
    place = 1
    for i in best:
        logging.info("  ------------------------------")
        logging.info("  Optimizing \'%s\' algorithm...", g_as_str(i))
        f = calc_f(i, a)
        name = os.path.splitext(filename)[0]
        plt.subplot(1, 3, place)
        optimize_energy(t, f, i, name)
        plt.plot(t, a, label="exp data")
        plt.title(g_as_str(i))
        plt.legend(loc="best")
        plt.draw()
        place += 1

        logging.info("  ------------------------------")
    plt.pause(7)
    name += ".png"
    plt.savefig(name, dpi=600)

"""
MAIN
"""
logging.info("===============RUNNING SCRIPT===============")

plt.figure(figsize=(18, 10))

if len(sys.argv) == 1:
    logging.info("No arguments, checking directory for .csv files...")
    for file in os.listdir():
        if file.endswith(".csv"):
            logging.info("===PROCESSING FILE===: %s", file)
            process_file(file)
            logging.info("==========DONE!==========")
if len(sys.argv) > 1:
    logging.info("Processing files from arguments...")
    for arg in sys.argv[1:]:
        logging.info("===PROCESSING FILE===: %s", arg)
        process_file(arg)
        logging.info("==========DONE!==========")
logging.info("================SHUTTING DOWN===============")
# End of script
