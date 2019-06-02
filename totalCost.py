import numpy as np
from myentropy import myentropy

def totalCost(CT):
  cost = 0

  #1-encode the CTs
  #1.1-sum code lengths
  for i in range(len(CT)):
    codeLens = CT[i][3]
    cost = cost + np.sum(codeLens[codeLens<np.inf])
  
  #1.2-fine labels encoding
  allels = []
  for i in range(len(CT)):
    els = CT[i][2]
    if type(els) is list:
      els = [flatten for inner in els for flatten in inner]
      #print([i.tolist() for i in els])
    allels.extend(list(els))
  e,_,_,_ = myentropy(allels)
  labelcost = len(allels)*e
  cost = cost + labelcost

  #2-encode the data
  cost_temp = cost
  for i in range(len(CT)):
    use = CT[i][4]
    codeLens = CT[i][3]
    idx = np.where(codeLens < np.inf)
    cost = cost + np.sum(codeLens[idx] * use[idx])

  return cost