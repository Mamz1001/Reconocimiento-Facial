predicción = 0.8
valor_real = 0
def error(predicción, valor_real):
    formula = (predicción - valor_real)**2
    print(formula)
error(predicción, valor_real)