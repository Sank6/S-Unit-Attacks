from tqdm import tqdm
from sage.all import *

import random, time, csv, copy
import numpy as np

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

def attack(n, sgens, alpha, structures):
    (K, OK, G, UK, units, embeddings) = structures
    s_units = units + sgens
    lm = log_lattice(K, units, sgens, embeddings)

    alpha_in = alpha

    log_alpha = log_embedding(K, alpha, units, sgens, embeddings)
    iterations = 0

    start = time.perf_counter()

    t = lm.solve_left(log_embedding(K, alpha, units, sgens, embeddings))
    rounded_t = [round(x) for x in t]
    powers = [0] + rounded_t[:-1]

    closest_element = prod([s_units[i] ** powers[i] for i in range(len(s_units))])
    unit_inverse = prod([s_units[i] ** -powers[i] for i in range(len(s_units))])
    rounded_norm = norm(vector(rounded_t[:-1]))
    real_norm = norm(t[:-1])
    new_alpha = alpha * unit_inverse
    new_alpha_norm = coefficient_norm(new_alpha)

    lowest_alpha = new_alpha if new_alpha_norm < coefficient_norm(alpha) else alpha
    lowest_alpha_norm = coefficient_norm(lowest_alpha)

    print(f"alpha: {lowest_alpha} ({lowest_alpha_norm})")

    for j in range(len(powers)):
        if powers[j] == 0: continue
        subbed_powers = copy.deepcopy(powers)
        subbed_powers[j] = 0

        closest_element = prod([s_units[i] ** subbed_powers[i] for i in range(len(s_units))])
        unit_inverse = prod([s_units[i] ** -subbed_powers[i] for i in range(len(s_units))])
        rounded_norm = norm(vector(rounded_t[:-1]))
        real_norm = norm(t[:-1])
        new_alpha = alpha * unit_inverse
        new_alpha_norm = coefficient_norm(new_alpha)
        
        if new_alpha_norm < lowest_alpha_norm:
            lowest_alpha = new_alpha
            lowest_alpha_norm = new_alpha_norm
            print(f"Powers: {subbed_powers}")
            print(f"New alpha: {new_alpha}")
            print(f"New alpha norm: {new_alpha_norm}")
    
    end = time.perf_counter()
    elapsed = end - start

    print(f"k: {len(sgens)},\nalpha_in: {alpha_in},\nalpha_in_norm: {coefficient_norm(alpha_in)},\nalpha_out: {lowest_alpha},\nalpha_out_norm: {coefficient_norm(lowest_alpha)},\ntime: {elapsed:.2f}s\n")

K.<x> = CyclotomicField(16)
OK = K.ring_of_integers()
G = K.galois_group()
UK = UnitGroup(K)
embeddings = K.embeddings(QQbar)
units = UK.gens_values()

alpha = OK.random_element()
print(f"alpha: {alpha} ({coefficient_norm(alpha)})")

y_bound = 100
sgens = list(dict.fromkeys(list(map(lambda x: x.gens_reduced()[0], K.primes_of_bounded_norm(y_bound)[1:]))))
print("SGens generated")
print(sgens)

attack(n, sgens, alpha, (K, OK, G, UK, units, embeddings))