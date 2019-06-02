import numpy as np
import pandas as pd

def myentropy(data):
  #ユニーク行の個数
  if np.array(data).ndim == 1:
    data = pd.Series(data)
    usages = np.array(data.groupby(data).size())
    els = np.sort(data.unique())
  else:
    data = pd.DataFrame(data)
    idx = list(data.columns)
    usages = np.array(data.groupby(idx).size())
    els = np.array(data.drop_duplicates())
  p = usages/np.sum(usages)

  bits = -np.log2(p)
  e = np.sum(p*bits)

  #els = np.sort(data.unique())

  return e, bits, els, usages