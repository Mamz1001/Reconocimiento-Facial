x = [0.8, 0.3, 0.9]
w = [ 0.5, 0.2, 0.8]
b = 0.1
u = 0
total = 0
for i in x:
    salida = x[u]*w[u]
    total += salida
    u += 1
total = total + b
print("Salida")
print(total)