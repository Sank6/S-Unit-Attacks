def log_embedding(K, alpha, n):
    embeddings = K.embeddings(QQbar)
    log_embeddings = []
    for i in range(n):
        sigma = embeddings[i]
        infinite_place = abs(sigma(alpha) * sigma(alpha).conjugate())
        log_embeddings.append(float(log(infinite_place)))
    return log_embeddings

coefficient_norm = lambda f: float(sqrt(sum([c.norm()**2 for c in f.list()])))

K.<x> = CyclotomicField(16)
R = K.ring_of_integers()

alpha = -23*x^7 - x^6 + 3*x^5 - 4*x^4 + 2*x^3 + 21 # R.random_element()
alpha_norm = coefficient_norm(alpha)
print(f"Alpha: {alpha} (norm = {alpha_norm:.2f})")

cyclotomic_units = [1+x+x^(-1), 1+x^3+x^(-3), 1+x^5+x^(-5)]
print(f"Units: {cyclotomic_units}")

n = len(cyclotomic_units) + 1

matrix_u = [log_embedding(K, unit, n) for unit in cyclotomic_units]
matrix_u.append([1 for _ in range(n)])
matrix_u = matrix(matrix_u)
print(f"Matrix M_U\n{matrix_u}")

y = log_embedding(K, alpha, n)
print(f"y: Log(alpha) = {y}")

t = matrix_u.solve_left(vector(y))
print(f"t: {t}")

rounded_t = [round(t[i]) for i in range(len(t))]
print(f"Rounded t: {rounded_t}")

corresponding_unit_inverse = K.one()
for i in range(n-1): corresponding_unit_inverse *= cyclotomic_units[i] ** (-rounded_t[i])
print(f"Corresponding unit inverse: {corresponding_unit_inverse}")

alpha_prime = alpha * corresponding_unit_inverse
alpha_prime_norm = coefficient_norm(alpha_prime)
print(f"Alpha': {alpha_prime} (norm = {alpha_prime_norm:.2f})")