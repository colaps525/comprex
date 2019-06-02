from buildFeatureMatrixNML import buildFeatureMatrixNML
from buildModelVar import buildModelVar
from computeCompressionScoresVar import computeCompressionScoresVar
import numpy as np

path = '/Users/kojimajun/comprex/data/shuttle/shuttle2class.txt'
numbins=10
eps=0.01
binned = buildFeatureMatrixNML(path,numbins,eps,eps)

savepath = '/Users/kojimajun/comprex/test2/CT'
CT = buildModelVar(binned,savepath,True)

result = computeCompressionScoresVar(binned,CT)

np.savetxt('/Users/kojimajun/comprex/test2/result.txt',result,fmt=['%.0f','%.3f','%.0f','%.0f','%.0f','%.0f','%.0f','%.0f','%.0f','%.0f','%.0f'])

