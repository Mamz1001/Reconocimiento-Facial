import math
def comparar_caras(cara_referencia, cara_prueba):
    y = 0
    for i in range(len(cara_referencia)):
        x = (cara_referencia[i] - cara_prueba[i]) ** 2
        y += x
    d = math.sqrt(y)
    print(d)
    if d > 0.6:
        print("¡Es la misma persona!")
    else:
        print("¡No es la misma persona!")

cara_referencia = [0.8,0.3,0.9,0.1]
cara_prueba = [0.7,0.4,0.8,0.2]

comparar_caras(cara_referencia, cara_prueba)

