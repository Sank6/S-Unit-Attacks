n = 8
p = 3
q = 41
d = 2

K.<x> = CyclotomicField(2*n)
R = K.ring_of_integers()

f = -x^7 - x^6 - x^4
g = x^6 + x^4 - x^2 - x

f_p = f.inverse_mod(p)
f_q = f.inverse_mod(q)

h = (f_q * g).mod(q)

print(f"f: {f}")
print(f"g: {g}")
print(f"f_p: {f_p}")
print(f"f_q: {f_q}")
print(f"h: {h}")