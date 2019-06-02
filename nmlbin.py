import numpy as np
import math
import copy

def NML_histogram(d,maxnumbins,eps,delta):
  def histogram_szpan(N,K):
    szpan = 0.0

    if N==0 or K==1:
      return 0.0

    szpan = szpan + ((K - 1.0)/2.0)*np.log(N/2.0)
    szpan = szpan + np.log(np.exp(math.lgamma(0.5)) / np.exp(math.lgamma(K/2.0)))
    szpan = szpan + (np.exp(math.lgamma(K/2.0))*K*np.sqrt(2.0))/(3.0*np.exp(math.lgamma(K/2.0-0.5))*np.sqrt(N))
    szpan = szpan + ((3.0+K*(K-2.0) * (2.0*K+1.0))/36.0-(np.exp(2.0*math.lgamma(K/2.0))*K*K)/(9.0*np.exp(2.0*math.lgamma((K/2.0-0.5)))))/N
    return szpan


  max_nof_bins = copy.copy(maxnumbins)
  epsilon = eps
  delta = delta

  d = np.array(d)
  N = d.size

  if delta < epsilon:
    print('!!!!!!!!!! Delta < epsilon !!!!!!!!!!')

  if len(d)==0:
    print('!!!!!!!!!! Could not find datafile !!!!!!!!!!')

  attr = np.floor(d*(1.0 / epsilon)+0.5)*epsilon
  attr = np.sort(attr)

  dmin = np.min(attr)
  dmax = np.max(attr)

  hist_min = dmin - (epsilon / 2.0)
  hist_max = dmax + (epsilon / 2.0)

  hist_range = (hist_max - hist_min) + 1e-10

  nof_pc = int(hist_range / delta) - 1

  if (nof_pc < 1):
    print('!!!!!!!!!! No potential cut points')

  potcut = np.zeros(nof_pc)
  H = np.zeros(nof_pc+1)
  for c in range(nof_pc):
    potcut[c] = hist_min + (c+1) * delta


  if (nof_pc+1) < max_nof_bins:
    max_nof_bins = nof_pc + 1

  j = 0
  for c in range(nof_pc):
    H[c] = j
    while (j < N) and (attr[j] < potcut[c]):
      H[c] = H[c] + 1
      j = j + 1
  H[nof_pc] = N

  #Dynamic programming starts
  BSC = np.zeros((nof_pc+1,max_nof_bins+1))
  BHG = np.zeros((nof_pc+1,max_nof_bins+1,nof_pc+1))

  HUGE_DOUBLE = 1E200
  for c in range(nof_pc+1):
    if nof_pc == 0:
      R = hist_max - hist_min
    else:
      R= (c+1) * delta

    BSC[c][0] = HUGE_DOUBLE
    BSC[c][1] = -1*H[c] * (np.log(H[c] * epsilon) - np.log(N*R))

  last = nof_pc
  bestnob = 1
  bestSC = BSC[last][1]
  for nob in range(2,max_nof_bins+1):
    for c in range(nob-1,nof_pc+1):
      minSC = HUGE_DOUBLE
      mintau = -1
      for tau in range(nob-2,c):
        if H[c] > H[tau]:
          if c == nof_pc:
            R = hist_max - potcut[tau]
          else:
            R = (c - tau)*delta

          SC = BSC[tau][nob-1] - (H[c]-H[tau])*(np.log((H[c]-H[tau])*epsilon) - np.log(N*R))
        elif H[c] == H[tau]:
          SC = BSC[tau][nob-1]
        else:
          print('!!!!!!!!!! H[c] < H[tau] !!!!!!!!!!')

        SC = SC + histogram_szpan(H[c],nob) - histogram_szpan(H[tau],nob-1)
        SC = SC + np.log((nof_pc-nob+2)/(nob-1))

        if(SC < minSC):
          minSC = SC
          mintau = tau

      BSC[c][nob] = minSC

      for s in range(nof_pc):
        BHG[c][nob][s] = BHG[mintau][nob-1][s]

      BHG[c][nob][mintau] = 1

    if BSC[last][nob] < bestSC:
      bestnob = nob
      bestSC = BSC[last][nob]

  cn = 1
  hg_cut = np.zeros(max_nof_bins+1)
  hg_cut[0] = hist_min

  for c in range(nof_pc):
    if BHG[last][bestnob][c] == 1:
      hg_cut[cn] = potcut[c]
      cn = cn + 1

  if cn != bestnob:
    print('!!!!!!!!!! cn != bestnob !!!!!!!!!!')

  hg_cut[bestnob] = hist_max
  hg_nof_bins = bestnob

  return hg_cut,hg_nof_bins

def nmlbin(d,maxnumbins,eps,delta,i):
  #Shg_cut,hg_nof_bins = NML_histogram(d,maxnumbins,eps,delta)
  d = np.array(d)
  N = d.size
  binned = np.zeros(N)-1

  hg_cut_test = np.array([[-1.005,-0.925,-0.915,-0.605,-0.595,-0.435,-0.425,-0.275,-0.265,0.635,2.765],
[-6.9950e+00,-2.7500e-01,-2.5000e-02,5.0000e-03,3.5000e-02,4.5000e-02,5.5000e-02,7.5000e-02,1.1875e+01,0.0000e+00,0.0000e+00],
[-1.255,-1.025,-1.015,-0.915,-0.905,-0.695,-0.685,-0.475,-0.465,0.315,3.095],
[-9.650e-01,-1.950e-01,-6.500e-02,-4.500e-02,-3.500e-02,-2.500e-02
,-1.500e-02,5.000e-03,1.500e-02,1.850e-01,5.175e+00],
[-2.585,0.205,0.215,0.385,0.395,0.475,0.485,0.565,0.575,1.025,1.565],
[-5.805,-0.215,-0.155,-0.075,-0.025,-0.015,0.045,0.135,0.195,1.685,10.655],
[-2.565,-0.905,0.155,0.165,0.225,0.235,0.305,0.315,0.455,0.465,2.645],
[-1.365,-1.185,-0.775,-0.765,-0.685,-0.675,-0.595,-0.585,-0.265,1.545,2.315],
[-1.025,-0.575,-0.565,-0.505,-0.495,-0.425,-0.415,-0.275,-0.265,0.785,1.755]])
  hg_cut = hg_cut_test[i]

  for c in range(int(hg_cut.size)-1):
    idx = np.where((d>=hg_cut[c]) & (d<hg_cut[c+1]))
    binned[idx] = c+1
  
  return binned




