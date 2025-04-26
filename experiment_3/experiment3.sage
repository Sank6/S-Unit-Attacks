from tqdm import tqdm
from sage.all import *

import random, time, csv, copy
import numpy as np

filename = f's_unit_attack_output_{time.strftime("%Y-%m-%d_%H.%M.%S")}.csv'

def log_embedding(K, f, units, sgens, embeddings):
    log_embeddings = []

    for i in range(len(units)):
        sigma = embeddings[i]
        infinite_norm = abs(sigma(f) * sigma(f).conjugate())
        log_embeddings.append(float(log(infinite_norm)))

    for i in range(len(sgens)):
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

def random(ring):
    alpha = ring.random_element()
    while coefficient_norm(alpha) < 1: alpha = ring.random_element()
    return alpha

def attack(n, sgens, alphas, writer, structures):
    (K, OK, G, UK, units, embeddings) = structures
    s_units = units + sgens
    lm = log_lattice(K, units, sgens, embeddings)

    for i in tqdm(range(len(alphas))):
        alpha = alphas[i]
        alpha_in = alpha

        log_alpha = log_embedding(K, alpha, units, sgens, embeddings)
        norm_is_large = True
        iterations = 0

        start = time.perf_counter()

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
        
        end = time.perf_counter()
        elapsed = end - start

        row = {
            'n': n,
            'k': len(sgens),
            'iteration': i + 1,
            'num_inner_iteration': iterations,
            'alpha_in': str(alpha_in),
            'alpha_in_norm': float(coefficient_norm(alpha_in)),
            'alpha_out': str(alpha),
            'alpha_out_norm': float(coefficient_norm(alpha)),
            'time': elapsed
        }
        writer.writerow(row)

if __name__ == '__main__':
    min_n = 3
    max_n = 20

    with open(filename, 'w', newline='') as csvfile:
        titles = ['n', 'k', 'iteration', 'num_inner_iteration','alpha_in', 'alpha_in_norm', 'alpha_out', 'alpha_out_norm', 'time']
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

            alphas = [random(OK) for _ in range(500)]
            
            y_bound = 150
            sgens = list(dict.fromkeys(list(map(lambda x: x.gens_reduced()[0], K.primes_of_bounded_norm(y_bound)[1:]))))

            print("Number of sgens: ", len(sgens))

            n_steps = 10
            step_size = max(floor(len(sgens) / n_steps), 1)
            for k in range(1, len(sgens), step_size):
                select_sgens = sgens[:k]
                attack(n, select_sgens, alphas, writer, (K, OK, G, UK, units, embeddings))