import matplotlib.pyplot as plt 

def plotData(X,y):
    plt.plot(X[(y==1)[:,0],1],X[(y==1)[:,0],2],'+',linewidth=2,markersize=7)
    plt.plot(X[(y==0)[:,0],1],X[(y==0)[:,0],2],'o',markersize=7)
    