n = 7
p = 3
q = 59
d = 2

K.<x> = CyclotomicField(n)
R = K.ring_of_integers()

f = 1 + x + x^3
g = x^3 + x^5

f_p = f.inverse_mod(p)
f_q = f.inverse_mod(q)

h = (p * f_q * g).mod(q)

print(f"f: {f}")
print(f"g: {g}")
print(f"f_p: {f_p}")
print(f"f_q: {f_q}")
print(f"h: {h}")