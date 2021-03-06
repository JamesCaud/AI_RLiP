import numpy as np
import matplotlib.pyplot as plt

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
        
def runExp(m1, m2, m3, epsilon, N):
    bandits = [MyBandit(m1), MyBandit(m2), MyBandit(m3)]
    
    data = np.empty(N)
    
    for i in range(N):
        # epsilon greedy
        p = np.random.random()
        if p < epsilon:
            j = np.random.choice(3)
        else:
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
    c_1 = runExp(1.0, 2.0, 3.0, .1, 100000)
    c_05 = runExp(1.0, 2.0, 3.0, .05, 100000) 
    c_01 = runExp(1.0, 2.0, 3.0, .01, 100000)
    
    # log scale plotting
    plt.plot(c_1, label='eps = .1')
    plt.plot(c_05, label='eps = .05')
    plt.plot(c_01, label='eps = .01')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()
