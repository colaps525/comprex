import numpy as np
import pandas as pd
from myentropy import myentropy
from totalCost import totalCost
import pickle
import copy

def buildModelVar(x,name,isavg):
  def add_CT(trace0, dimensions,nomrows,elementList,codeLenList,usages):
    trace_tmp = []
    trace_tmp.append(dimensions)
    trace_tmp.append(nomrows)
    trace_tmp.append(elementList)
    trace_tmp.append(codeLenList)
    trace_tmp.append(usages)
    trace0.append(trace_tmp)

    return trace0

  def array_flatten(els):
    items = []
    for v in els:
      if isinstance(v, (np.float64, int) ):
        items.append(v)
      else:
        for item in list(v):
          items.append(item)
    return items

  def array_of_array(els):
    items = []
    for v in els:
      if isinstance(v, (np.float64, int,float) ):
        items.append(np.array([v]))
      else:
        items.append(v)
    return items

  def checkiftried(newgroup,triedgroups):
    iscovered = False
    newgroup = array_flatten(newgroup)
    for tr in range(len(triedgroups)):
      if len(newgroup) != len(triedgroups[tr]):
        continue
      ai = set(newgroup).intersection(set(array_flatten(triedgroups[tr])))
      if len(ai) == len(newgroup):
        iscovered = True
        break
    return iscovered

  def isincover(dd,elj):
    C = set(dd).intersection(set(elj))
    #if list(C) in list(dd):
    if len(list(C)) == len(elj):
      covers = True
      dd = set(dd).difference(C)
      dd = np.array(list(dd))
      #ai = list(dd).index(list(C))
    else:
      covers = False
      #ai = -1

    #if ai > -1:
    #  covers = True
      ##dd[ai] = []
    #  dd = list(dd)
    #  del dd[ai]
    #  dd = np.array(dd)
    #else:
    #  covers = False
    return dd,covers


  #x = np.loadtxt('/Users/kojimajun/comprex/data/shuttle/np_savetxt.txt')
  x = x
  f = x.shape[1]

  CT = []
  for i in range(f):
    #codeLens:bits
    e, codeLens, elements, usages = myentropy(x[:,i])
    CT = add_CT(CT,i,len(elements),elements,codeLens,usages)

  cost = totalCost(CT)

  print('************************************')
  print('Elementary total cost: ',cost)
  print('Number of CTs: ',len(CT))
  print('************************************')


  totalIGtime = 0
  #compute Information-Gain matrix of features for merging candidates
  triedgroups = []
  while 1:
    if len(CT) < 2:
      print('************************************')
      print('Elementary total cost: ',cost)
      print('Number of CTs: ',len(CT))
      print('************************************')
      with open(name,'wb') as f:
        pickle.dump(CT,f)
      break

    #first round, build elementary IGs
    if len(CT) == f:
      IG = np.zeros((len(CT),len(CT)))
      if isavg:
        Fsize = np.zeros((len(CT),len(CT)))
      for i in range(len(CT)):
        ei,_,_,_ = myentropy(x[:,CT[i][0]])
        for j in range(i+1,len(CT)):
          ej,_,_,_ = myentropy(x[:,CT[j][0]])
          eij,_,_,_ = myentropy(x[:,[CT[i][0],CT[j][0]]])
          IG[i,j] = ei + ej - eij
          if isavg:
            Fsize[i,j] = 2
    #somethins is merged,update
    #find index of non-merged sets
    else:
      print('************************************')
      print('Elementary total cost: ',cost)
      print('Number of CTs: ',len(CT))
      print('************************************')

      allsets = np.arange(len(IG))
      ind = np.where((allsets != ct1) & (allsets != ct2))
      IG = IG[:,ind]
      IG = IG[ind,:]
      IG = IG.reshape([len(ind[0]),len(ind[0])])
      IG = np.hstack([IG,np.zeros((len(ind[0]),1))])
      IG = np.vstack([IG,np.zeros((1,len(ind[0])+1))])

      if isavg:
        Fsize = Fsize[:,ind]
        Fsize = Fsize[ind,:]
        Fsize = Fsize.reshape([len(ind[0]),len(ind[0])])
        Fsize = np.hstack([Fsize,np.zeros((len(ind[0]),1))])
        Fsize = np.vstack([Fsize,np.zeros((1,len(ind[0])+1))])

      ej,_,_,_ = myentropy(x[:,CT[-1][0]])
      for i in range(len(CT)-1):
        ei,_,_,_ = myentropy(x[:,CT[i][0]])
        tmp = np.hstack([CT[i][0],CT[-1][0]])
        eij,_,_,_ = myentropy(x[:,tmp])
        IG[i,-1] = ei + ej-eij
        if isavg:
          Fsize[i,-1] = len(np.array([CT[i][0]]).flatten()) + len(np.array([CT[-1][0]]).flatten())

    ig = IG[IG != 0]
    ct1, ct2 = np.where(IG != 0)

    if len(ig) == 0:
      print('************************************')
      print('Elementary total cost: ',cost)
      print('Number of CTs: ',len(CT))
      print('************************************')
      with open(name,'wb') as f:
        pickle.dump(CT,f)
      break

    if isavg:
      ridx = list(np.unique(ct1))
      cidx = list(np.unique(ct2))
      Fsizetmp = Fsize[ridx,:]
      Fsizetmp = Fsizetmp[:,cidx]
      fsize = Fsizetmp[Fsizetmp != 0]
      ig = ig / fsize

    anymerged = False

    sig = np.sort(ig)[::-1]
    ix = np.argsort(ig)[::-1]
    ctt1 = ct1[ix]
    ctt2 = ct2[ix]

    #try to merge those groups with highest information gain in order
    for ct in range(len(ct1)):
      ct1 = ctt1[ct]
      ct2 = ctt2[ct]
      print('Trying to merge groups',ct1,'and',ct2)
      newgroup = [CT[ct1][0],CT[ct2][0]]
      if checkiftried(newgroup,triedgroups):
        print('... has been tried before ...')
        continue
      else:
        triedgroups.append(newgroup)

      tempCT = copy.deepcopy(CT)
      del tempCT[ct1]
      del tempCT[ct2-1]

      ########## build new CT ############
      newdims = [CT[ct1][0],CT[ct2][0]]
      newdims = array_flatten(newdims)

      #put all the elements from both tables together
      newels = CT[ct1][2]
      newusages = CT[ct1][4]

      newusages = np.hstack((newusages,CT[ct2][4]))
      newels = np.hstack((newels,CT[ct2][2]))

      #newels = np.array([[i] for i in newels])
      newels = np.array(array_of_array(newels))
      ellens = [len(i) for i in newels]

      tmp_sort = np.hstack((ellens,newusages)).reshape(2,len(ellens))
      ind = list(np.argsort(tmp_sort[1]))
      newels = newels[ind]
      newels = list(newels)
      newusages = newusages[ind]

      fx = x[:,newdims]
      fx = pd.DataFrame(fx)
      #uq = np.array(fx.drop_duplicates())
      tmp_idx = list(fx.columns)
      counts = np.array(fx.groupby(tmp_idx).size())
      uq = fx.groupby(tmp_idx).size()
      uq = np.array(uq.index.values.tolist())
      ix = np.argsort(counts)[::-1]
      scounts = counts[ix]
      urows = uq[ix]

      numnoreduce = 0
      temp_cost_prev = 0
      U = len(urows)
      print('Number of unique row on newdims: ',U)
      newels_before_u = len(newels)

      for u in range(U):
        newels.append(urows[u,:])
        newusages = np.hstack((newusages,scounts[u]))

        ##########################################
        #update usages for those overlapping one
        ##########################################
        dd = newels[-1]
        for k in reversed(range(newels_before_u)):
          dd,covers = isincover(dd,newels[k])
          if covers:
            newusages[k] = newusages[k] - newusages[-1]
          if newusages[k] <= -1:
            print('ERROR assert newusages[k] <= -1')
          if len(dd) == 0:
            break

        ###########################################
        #find >1 length elements with usage 0 and remove them
        ###########################################
        lens = np.array([len(i) for i in newels])
        ind_bool = (lens > 1)*(newusages ==0)
        ind = np.where(ind_bool)[0]

        if len(ind) != 0:
          newusages = newusages.tolist()
          lens = lens.tolist()
          for i in ind:
            del newels[i]
            del newusages[i]
            del lens[i]
          newusages = np.array(newusages)
          lens = np.array(lens)

        tmp_sort = np.hstack((lens,newusages)).reshape(2,len(lens))
        ind = list(np.lexsort((tmp_sort[1],tmp_sort[0])))
        newels = [newels[i] for i in ind]
        newusages = newusages[ind]

        newnumrows = len(newels)

        #given usages, find codeLens
        newCodeLens = np.zeros(len(newusages)) + np.inf
        ind = list(np.where(newusages > 0)[0])
        newCodeLens[ind] = -np.log2(newusages[ind]/np.sum(newusages))

        #now add the new CT to the rest
        #print('tempCT',tempCT)
        #print('newdims',newdims)
        #print('newnumrows',newnumrows)
        #print('newels',newels)
        #print('newCodeLens',newCodeLens)
        #print('newusages',newusages)
        tempCT = add_CT(tempCT,newdims,newnumrows,newels,newCodeLens,newusages)

        #now check the new total cost
        temp_cost = totalCost(tempCT)

        if np.round(temp_cost) >= np.round(temp_cost_prev):
          numnoreduce = numnoreduce + 1
        else:
          numnoreduce = max([0,numnoreduce-1])

        if temp_cost < cost:
          print('Inserting up to row ',u,' w/ freq. ',scounts[u],' reduced total cost: ',cost,' vs ',temp_cost)
          CT = copy.deepcopy(tempCT)
          cost = temp_cost
          anymerged = True

        #if it has been increasing for so many times consecutively
        if numnoreduce == 5:
          break

        #remove the CT from the end
        del tempCT[-1]

        temp_cost_prev = temp_cost
        #print(dm)
      if anymerged:
        print('Merge ',ct1,' - ',ct2,' feasible, cost: ',cost, ', #CTs: ',len(CT))
        break

    if not anymerged:
      print('************************************')
      print('Final total cost: ',cost)
      print('Number of CTs: ',len(CT))
      print('************************************')
      with open(name,'wb') as f:
        pickle.dump(CT,f)
      break
  return CT



if __name__ == '__main__':
  buildModelVar('test','test',True)
