def ReLU(x): 
    filtro = max(0, x)
    if filtro > 0:
        print("positivo: ¡pasa!")
    elif filtro < 0:
        print("negativo: bloqueado")
    else:
        print("cero: ¡tambien bloqueado!")

print(ReLU(1.28))
print(ReLU(-0.5))
print(ReLU(0))

