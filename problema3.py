import math
def sigmoid(x):
    total = 1 / (1 + math.exp(-x))
    if total > 0.5:
        print(total*100,"% seguro que SI")
    elif total < 0.5:
        print(total*100,"% seguro que NO")
    else:
        print(total*100,"% un poco si y un poco no")


print(sigmoid(2))