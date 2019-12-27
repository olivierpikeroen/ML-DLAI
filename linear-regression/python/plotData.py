import matplotlib.pyplot as plt

def plotData(x,y):
    """
    Plots the data points x and y into a new figure 
    
    Plots the data points and gives the figure axes labels of population and profit.
    """
    line,=plt.plot(x,y,'rx')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    return line