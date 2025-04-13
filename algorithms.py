import numpy as np
from random import randint

def gram_schmidt(basis):
    n = len(basis)
    ortho_basis = []
    projection_coeff = np.zeros((n, n)) # mu

    for i in range(n):
        new_vec = np.array(basis[i], dtype=np.float64)
        for j, vec in enumerate(ortho_basis):
            projection_coeff[i][j] = np.dot(new_vec, vec) / np.dot(vec, vec)
            projection = projection_coeff[i][j] * vec
            new_vec -= projection
        ortho_basis.append(new_vec)

    return ortho_basis, projection_coeff

def LLL(basis, delta=0.75):
    basis = [np.array(vec, dtype=np.float64) for vec in basis]
    n = len(basis)
    ortho_basis, projection_coeff = gram_schmidt(basis)
    k = 1

    while k < n:
        # Size reduction
        for j in reversed(range(k)):
            mu = round(projection_coeff[k][j])
            if mu != 0:
                basis[k] -= mu * basis[j]
                ortho_basis, projection_coeff = gram_schmidt(basis)

        # Lovasz condition
        lhs = np.dot(ortho_basis[k], ortho_basis[k])
        rhs = (delta - projection_coeff[k][k - 1] ** 2) * np.dot(ortho_basis[k - 1], ortho_basis[k - 1])

        if lhs >= rhs: k += 1
        else:
            basis[k], basis[k - 1] = basis[k - 1], basis[k]
            ortho_basis, mu = gram_schmidt(basis)
            k = max(k - 1, 1)

    return [vec.astype(np.int64) for vec in basis]

def babai_roundoff(basis, target):
    basis = [np.array(vec, dtype=np.float64) for vec in basis]
    target = np.array(target, dtype=np.float64)
    ortho_basis, _ = gram_schmidt(basis)
    n = len(basis)

    y = target.copy()
    coeffs = np.zeros(n)

    for i in reversed(range(n)):
        t = np.dot(y, ortho_basis[i]) / np.dot(ortho_basis[i], ortho_basis[i])
        coeffs[i] = round(t)
        y -= coeffs[i] * basis[i]

    closest_vec = sum(int(c) * b for c, b in zip(coeffs, basis))
    return closest_vec.astype(np.int64)

if __name__ == "__main__":
    basis = [[randint(-10, 10) for _ in range(3)] for _ in range(3)]
    print("Original Basis:")
    for vec in basis: print(vec)
    
    print("\nGram-Schmidt Orthogonalization:")
    ortho_basis, mu = gram_schmidt(basis)
    for vec in ortho_basis: print(vec)
    
    print("\nProjection Coefficients:")
    for row in mu: print(row)

    reduced_basis = LLL(basis)
    print("\nLLL Reduced Basis:")
    for vec in reduced_basis: print(vec)
    
    target = [randint(-10, 10) for _ in range(3)]
    print("\nTarget Vector:", target)
    closest_vec = babai_roundoff(reduced_basis, target)
    print("Closest Vector:", closest_vec)
