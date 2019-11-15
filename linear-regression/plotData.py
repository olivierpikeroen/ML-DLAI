import matplotlib.pyplot as plt

def plotData(x,y):
    line,=plt.plot(x,y,'rx')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    return line