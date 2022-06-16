import matplotlib.pyplot as plt
from numpy import ndarray
def plot_scores(scores:ndarray=None,score_name:str='Accuracy'):
  """
  Plot accuracy during training
  """
  plt.plot(list(range(1,len(scores)+1)),scores,'--')
  plt.xlabel('Iterations')
  plt.ylabel('{}'.format(score_name))
  plt.show()
  
