from tqdm import tqdm
from sage.all import *

import random, time, csv
import numpy as np

filename = f's_unit_attack_output_{time.strftime("%Y-%m-%d_%H.%M.%S")}.csv'

def log_embedding(K, f, units, sgens, embeddings):
    log_embeddings = []

    for i in range(0, len(units)):
        sigma = embeddings[i]
        infinite_norm = abs(sigma(f) * sigma(f).conjugate())
        log_embeddings.append(float(log(infinite_norm)))

    for i in range(0, len(sgens)):
        finite_norm = sgens[i].norm() ** (-K.valuation(sgens[i])(f))
        log_embeddings.append(float(log(finite_norm)))

    return vector(log_embeddings)

def log_lattice(K, units, sgens, embeddings):
    log_lattice_out = []
    for f in units[1:] + sgens:
        log_lattice_out.append(log_embedding(K, f, units, sgens, embeddings))

    log_lattice_out.append([1 for i in range(len(units) + len(sgens))])  # Final row of 1s
    return matrix(log_lattice_out)

def coefficient_norm(f):
    coefficients = f.list()
    return float(sqrt(sum([c.norm()**2 for c in coefficients])))

def attack(n, y, writer, structures):
    (K, OK, G, UK, units, embeddings) = structures
    sgens = list(dict.fromkeys(list(map(lambda x: x.gens_reduced()[0], K.primes_of_bounded_norm(y)[1:]))))
    s_units = units + sgens
    lm = log_lattice(K, units, sgens, embeddings)

    for i in tqdm(range(0, 500)):
        alpha = OK.random_element()
        while coefficient_norm(alpha) < 1: alpha = OK.random_element()
        alpha_in = alpha

        log_alpha = log_embedding(K, alpha, units, sgens, embeddings)
        norm_is_large = True
        iterations = 0

        while norm_is_large and iterations < 100:
            iterations += 1
            t = lm.solve_left(log_embedding(K, alpha, units, sgens, embeddings))
            rounded_t = [round(x) for x in t]
            powers = [0] + rounded_t[:-1]

            closest_element = prod([s_units[i] ** powers[i] for i in range(len(s_units))])
            unit_inverse = prod([s_units[i] ** -powers[i] for i in range(len(s_units))])
            rounded_norm = norm(vector(rounded_t[:-1]))
            real_norm = norm(t[:-1])
            new_alpha = alpha * unit_inverse
            new_alpha_norm = coefficient_norm(new_alpha)

            if new_alpha_norm < coefficient_norm(alpha): alpha = new_alpha
            else: break
        
        row = {
            'n': n,
            'y': y,
            'iteration': i + 1,
            'num_inner_iteration': iterations,
            'alpha_in': str(alpha_in),
            'alpha_in_norm': float(coefficient_norm(alpha_in)),
            'alpha_out': str(alpha),
            'alpha_out_norm': float(coefficient_norm(alpha)),
        }
        writer.writerow(row)

if __name__ == '__main__':
    min_n = 3
    max_n = 20

    with open(filename, 'w', newline='') as csvfile:
        titles = ['n', 'y', 'iteration', 'num_inner_iteration','alpha_in', 'alpha_in_norm', 'alpha_out', 'alpha_out_norm']
        writer = csv.DictWriter(csvfile, fieldnames=titles)
        writer.writeheader()

        for n in range(min_n, max_n + 1, 1):
            print("Running for n = ", n)
            K.<x> = CyclotomicField(n)
            OK = K.ring_of_integers()
            G = K.galois_group()
            UK = UnitGroup(K)
            embeddings = K.embeddings(QQbar)
            units = UK.gens_values()

            min_y_val = max(1, n - 5)
            max_y_val = n + 20

            for y in range(min_y_val, max_y_val + 1, 1):
                attack(n, y, writer, (K, OK, G, UK, units, embeddings))