import pickle
import numpy as np

def computeCompressionScoresVar(datafile,CTfile):
  def isincover(dd,elj):
    if isinstance(dd,(np.float64)):
      dd = [dd]
    if isinstance(elj,(np.float64)):
      elj = [elj]

    C = set(dd).intersection(set(elj))
    if len(list(C)) == len(elj):
      covers = True
      dd = set(dd).difference(C)
      dd = np.array(list(dd))
    else:
      covers = False

    return dd,covers

  #with open(CTfile, 'rb') as f:
  #  CT= pickle.load(f)

  #data = np.loadtxt(datafile)
  CT = CTfile
  data = datafile
  f = data.shape[1]
  N = data.shape[0]

  scores = np.zeros(N)
  
  for i in range(N):
    d = data[i,:]

    for j in range(len(CT)):
      dims = CT[j][0]
      dd = d[dims]

      els = CT[j][2]
      codeLens = CT[j][3]

      for k in reversed(range(len(els))):
        dd,covers = isincover(dd, els[k])
        if covers:
          scores[i] = scores[i] + codeLens[k]
          if not (scores[i] < np.inf):
            print('!!!!! scors[i] is inf !!!!!')
          if len(dd) == 0:
            break

  #ind = np.argsort(scores)[::-1]
  ind = np.argsort(scores)[::-1].reshape([data.shape[0],1])
  sscores = np.sort(scores)[::-1].reshape([data.shape[0],1])
  #sscores = np.sort(scores)[::-1]
  sdata = data[ind[:,0],:]

  result = np.hstack([ind,sscores,sdata])
  
  #np.savetxt('/Users/kojimajun/comprex/data/shuttle/result.txt',result,fmt=['%.0f','%.3f','%.0f','%.0f','%.0f','%.0f','%.0f','%.0f','%.0f','%.0f','%.0f'])
  return result



if __name__ == '__main__':
  CTfile = '/Users/kojimajun/comprex/data/shuttle/CT'
  datafile = '/Users/kojimajun/comprex/data/shuttle/np_savetxt.txt'
  computeCompressionScoresVar(datafile,CTfile)