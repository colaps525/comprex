import pandas as pd
import numpy as np
from nmlbin import nmlbin

def buildFeatureMatrixNML(filename,maxnumbins,epsilon,delta):
  data = pd.read_csv(filename,header=None,delimiter=' ')
  
  row_num,col_num = data.shape
  binned = np.zeros((row_num,col_num))
  #print(row_num,col_num)

  for i in range(col_num):
    d = data.iloc[:,i]
    binned[:,i] = nmlbin(d, maxnumbins,epsilon,delta,i)
  #print(binned)

  for k in range(1,col_num):
    print(k)
    #列ごとにインデックスの重複がないように、前列の最大値からインデックスを振る
    binned[:,k] = np.max(binned[:,k-1])+binned[:,k]
  #print(binned)
  #np.savetxt('/Users/kojimajun/comprex/data/shuttle/np_savetxt.txt', binned, fmt='%d')

  return binned

if __name__ == '__main__':
  path = '/Users/kojimajun/comprex/data/shuttle/shuttle2class.txt'
  numbins=10
  eps=0.01
  buildFeatureMatrixNML(path,numbins,eps,eps)