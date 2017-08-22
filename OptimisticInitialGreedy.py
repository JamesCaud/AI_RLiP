import numpy as np
import matplotlib.pyplot as plt
from EpsilonGreedy import runExp as runExpEps

class MyBandit:
    def __init__(self, m, upperLimit):
        self.m = m
        # different
        self.mean = upperLimit
        self.N = 0
    
    def pull(self): 
        return np.random.randn() + self.m
    
    def update(self, x):
        self.N += 1
        # (previous mean * % of new mean) + (new val * % of new mean)
        self.mean = ((1 - 1.0/self.N) * self.mean) + (1.0/self.N * x)
        
def runExp(m1, m2, m3, N, upperLimit = 10):
    bandits = [MyBandit(m1, upperLimit), MyBandit(m2, upperLimit), MyBandit(m3, upperLimit)]
    
    data = np.empty(N)

    # different
    for i in range(N):
        j = np.argmax([b.mean for b in bandits])
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
    
    for b in bandits:
        print (b.mean)
        
    return cumAvg
    
if __name__ == '__main__':
  c_1 = runExpEps(1.0, 2.0, 3.0, 0.1, 100000)
  oiv = runExp(1.0, 2.0, 3.0, 100000)

  # log scale plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(oiv, label='optimistic')
  plt.legend()
  plt.xscale('log')
  plt.show()


  # linear plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(oiv, label='optimistic')
  plt.legend()
  plt.show()
