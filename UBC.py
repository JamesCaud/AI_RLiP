import numpy as np
import matplotlib.pyplot as plt
from EpsilonGreedy import runExp as runExpEps

class MyBandit:
    def __init__(self, m):
        self.m = m
        self.mean = 0
        self.N = 0
    
    def pull(self): 
        return np.random.randn() + self.m
    
    def update(self, x):
        self.N += 1
        # (previous mean * % of new mean) + (new val * % of new mean)
        self.mean = ((1 - 1.0/self.N) * self.mean) + (1.0/self.N * x)
        
def ubc(mean, n, nj):
    if nj == 0:
        return float('inf')
    return mean + np.sqrt(2*np.log(n) / nj)
        
def runExp(m1, m2, m3, N, upperLimit = 10):
    bandits = [MyBandit(m1), MyBandit(m2), MyBandit(m3)]
    
    data = np.empty(N)

    # different
    for i in range(N):
        j = np.argmax([ucb(b.mean, i+1, b.N) for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        
        # for plotting
        data[i] = x
    cumAvg = np.cumsum(data) / (np.arange(N) + 1)
    
    # plot moving average
    plt.plot(cumAvg)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()
    
    # for b in bandits:
    #    print (b.mean)
        
    return cumAvg
    
if __name__ == '__main__':
  c_1 = runExpEps(1.0, 2.0, 3.0, 0.1, 100000)
  ucb = runExp(1.0, 2.0, 3.0, 100000)

  # log scale plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(ucb, label='ucb')
  plt.legend()
  plt.xscale('log')
  plt.show()


  # linear plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(ucb, label='ucb')
  plt.legend()
  plt.show()
