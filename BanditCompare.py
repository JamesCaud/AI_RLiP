import numpy as np
import matplotlib.pylot as plt
from EpsilonGreedy import MyBandit as Bandit
from OptimisticInitialGreedy import runExp as runExpOIV
from UBC import runExp as runExpUCB

class BayesianBandit:
  def __init__(self, trueMean):
    self.trueMean = trueMean
    # mu parameters
    # prior is N(0,1)
    self.predictedMean = 0
    self.lambda_ = 1
    self.sum_x = 0
    self.tau = 1
    
  def pull(self):
    return np.random.randn() + self.trueMean
    
  def sample(self):
    return np.random.randn() / np.sqrt(self.lambda_) + self.predictedMean
    
  def update(self, x):
    self.lambda_ += 1
    self.sum_x += x
    self.predictedMean = self.tau * self.sum_x / self.lambda_
    
def runExpDecayEpsilon(m1, m2, m3, N):
  bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
  
  data = np.empty(N)
  
  for i in range(N):
    # epsilon greedy
    p = np.random.random()
    if p < 1.0/(i+1):
      j = np.random.choice(3)
    else:
      j = np.argmax([b.mean for b in bandits])
    x = bandits[j].pull()
    bandits[j].update(x)
    
    data[i] = x
    
  cumAvg = np.cumsum(data) / (np.arange(N) + 1)
  
  # plot moving average ctr
  plt.plot(cumAvg)
  plt.plot(np.ones(N)*m1)
  plt.plot(np.ones(N)*m2)
  plt.plot(np.ones(N)*m3)
  plt.xscale('log')
  plt.show()

  for b in bandits:
    print(b.mean)

  return cumAvg
  
def runExp(m1, m2, m3, N):
  bandits = [BayesianBandit(m1), BayesianBandit(m2), BayesianBandit(m3)]

  data = np.empty(N)
  
  for i in range(N):
    j = np.argmax([b.sample() for b in bandits])
    x = bandits[j].pull()
    bandits[j].update(x)

    data[i] = x
  cumAvg = np.cumsum(data) / (np.arange(N) + 1)

  # plot moving average ctr
  plt.plot(cumAvg)
  plt.plot(np.ones(N)*m1)
  plt.plot(np.ones(N)*m2)
  plt.plot(np.ones(N)*m3)
  plt.xscale('log')
  plt.show()

  return cumAvg

if __name__ == '__main__':
  m1 = 1.0
  m2 = 2.0
  m3 = 3.0
  eps = runExpDecayEpsilon(m1, m2, m3, 100000)
  oiv = runExpOIV(m1, m2, m3, 100000)
  ucb = runExpUCB(m1, m2, m3, 100000)
  bayes = runExp(m1, m2, m3, 100000)

  # log scale plot
  plt.plot(eps, label='decaying-epsilon-greedy')
  plt.plot(oiv, label='optimistic')
  plt.plot(ucb, label='ucb1')
  plt.plot(bayes, label='bayesian')
  plt.legend()
  plt.xscale('log')
  plt.show()


  # linear plot
  plt.plot(eps, label='decaying-epsilon-greedy')
  plt.plot(oiv, label='optimistic')
  plt.plot(ucb, label='ucb1')
  plt.plot(bayes, label='bayesian')
  plt.legend()
  plt.show()
