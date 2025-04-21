def log_embedding(K, f, n, sgens):
    embeddings = K.embeddings(QQbar)
    log_embeddings = []
    for i in range(n):
        sigma = embeddings[i]
        infinite_norm = abs(sigma(f) * sigma(f).conjugate())
        log_embeddings.append(float(log(infinite_norm)))

    for i in range(len(sgens)):
        finite_norm = sgens[i].norm() ** (-K.valuation(sgens[i])(f))
        log_embeddings.append(float(log(finite_norm)))

    return log_embeddings

coefficient_norm = lambda f: float(sqrt(sum([c.norm()**2 for c in f.list()])))

K.<x> = CyclotomicField(16)
R = K.ring_of_integers()

alpha = -12*x^7 + 5*x^6 - 17*x^5 - 27*x^4 + 26*x^3 - 8*x^2 + 26*x - 7 # R.random_element()
alpha_norm = coefficient_norm(alpha)
print(f"Alpha: {alpha} (norm = {alpha_norm:.2f})")

UK = K.unit_group()
units = UK.gens_values()
n = len(units)
print(f"Units: {units}")

y_bound = 30
sgens = list(dict.fromkeys(list(map(lambda x: x.gens_reduced()[0], K.primes_of_bounded_norm(y_bound)[1:]))))
print(f"S-Unit Generators: {sgens} (Count: {len(sgens)})")

matrix_s = [log_embedding(K, f, n, sgens) for f in units[1:]]
for f in sgens: matrix_s.append(log_embedding(K, f, n, sgens))
matrix_s.append([1 for _ in range(n + len(sgens))])
matrix_s = matrix(matrix_s)

y = log_embedding(K, alpha, n, sgens)
print(f"y: Log(alpha) = {y}")

t = matrix_s.solve_left(vector(y))
print(f"t: {t}")

rounded_t = [round(x) for x in t]
print(f"Rounded t: {rounded_t}")

powers = [0] + rounded_t[:-1]
s_units = units + sgens
corresponding_s_unit_inverse = prod([s_units[i] ** (-powers[i]) for i in range(len(s_units))])
print(f"Corresponding S-unit inverse: {corresponding_s_unit_inverse}")

alpha_prime = alpha * corresponding_s_unit_inverse
alpha_prime_norm = coefficient_norm(alpha_prime)
print(f"Alpha': {alpha_prime} (norm = {alpha_prime_norm:.2f})")