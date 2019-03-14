import sys, os, glob, cv2, math
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import externals
import scipy.stats as ss
import seaborn as sns


class needlePassing:
    def __init__(self, dataPath):
        np.seterr(divide='ignore', invalid='ignore')
        self.dataPath = dataPath
        self.plotmultivariateDistributions("cartesian")# use either "cartesian" or "rotation" for input to the function

    def loadOffsets(self):
        dfLength = 78
        kinOffset = {}
        kinSpan = {}
        kinOffset['cartesian'] = 0
        kinOffset['rotation'] = 3
        kinOffset['linearVelocity'] = 12
        kinOffset['angularVelocity'] = 15
        kinOffset['grasperAngle'] = 18
        kinSpan['cartesian'] = 3
        kinSpan['rotation'] = 9
        kinSpan['linearVelocity'] = 3
        kinSpan['angularVelocity'] = 3
        kinSpan['grasperAngle'] = 1
        return kinOffset, kinSpan

    def plotmultivariateDistributions(self, key1):
        """
        This function plots plots the KL-divergence for cartesian or orientation, by loading the annotated segment-wise trajectories
        """
        kinOffset, kinSpan = self.loadOffsets()
        errorFile = self.dataPath + "/segments/errorcsvs/{}*.csv".format(key1)
        kl_dict = {}
        aggregate_dict= {}
        for name in glob.glob(errorFile):
            _gesture = name.split("/")[len(name.split("/"))-1]
            cartesians = np.asarray(pd.read_csv(name))
            if key1 == "rotation":
                orientation = self.orientation(cartesians[:,1:cartesians.shape[1]-1])
            else:
                print ("non-rotation")
                orientation = (cartesians[:,1:cartesians.shape[1]-1])
            print (_gesture)
            suboptimal_cart = []
            optimal_cart = []
            for i in range(cartesians.shape[0]):
                if cartesians[i][cartesians.shape[1]-1] == 1:
                    suboptimal_cart.append(orientation[i])
                else:
                    optimal_cart.append(orientation[i])
            suboptimal_cart =  (np.asarray(suboptimal_cart))
            hist_list = []
            optimal_cart = (np.asarray(optimal_cart))
            labelKeys = self.getLabels(key1)
            manip = "left"
            subplots = 3
            subplotnum1 =int("{}1".format(subplots))
            _color = ['red', 'white']
            dists = [optimal_cart, suboptimal_cart]
            histList = []
            _maxlist = []
            _minlist = []
            print (suboptimal_cart.shape)

            for i in range(optimal_cart.shape[1]):
                _minlist.append(min(min(optimal_cart[:,i]),min(suboptimal_cart[:,i])))
                _maxlist.append(max(max(optimal_cart[:,i]),max(suboptimal_cart[:,i])))

            for count in range(len(dists)):
                k = 0
                manip = "left"
                for k in range(0,2):
                    if k>0:
                        manip = "right"
                    flattened_array = dists[count][:,k*3+0:k*3+3]
                    num = 100
                    H, edges = np.histogramdd(flattened_array, bins = (num, num, num), range= [[_minlist[k*3+0], _maxlist[k*3+0]], [_minlist[k*3+1],_maxlist[k*3+1]], [_minlist[k*3+2],_maxlist[k*3+2]]])
                    #self.surfacePlots(dists[count][:,k*3+0],dists[count][:,k*3+1],dists[count][:,k*3+2], H, edges)
                    hist_list.append(H)
            for j1 in range(len(hist_list)):
                for val in (hist_list[j1]):
                    zero_indices = np.transpose(np.nonzero(val==0))
                    for index in zero_indices:
                        val[index[0]][index[1]] = 0.00000000000001
                #print (hist_list[j1])

            kl_dict[_gesture]= []
            aggregate_dict[_gesture] = 0.0
            for j1 in range (0,len(hist_list)/2):
                kl_dict[_gesture].append(ss.entropy(hist_list[j1], hist_list[j1+len(hist_list)/2])) # compares the first with the 6th, 2nd with 7th and so on
            aggregate_dict[_gesture] = sum(kl_dict[_gesture])/len(kl_dict[_gesture])
        externals.joblib.dump(aggregate_dict, 'aggregate_dict.p')
        aggregate_dict = externals.joblib.load('aggregate_dict.p')
        lists = sorted(aggregate_dict.items()) # sorted by key, return a list of tuples
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        x = list(x)
        for i in range(len(x)):
            x[i] = x[i].replace(key1+"s", "").replace(".p","").replace('.csv','')
        sum_y = np.zeros(len(x))
        for i in range(len(y)):
            sum_y[i] = np.sum(y[i])
        plt.plot(x, sum_y)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=25)
        plt.xlabel("Gesture Index", fontsize=30)
        plt.ylabel("Entropy", fontsize=30)
        plt.show()

    def surfacePlots(self, X, Y, Z, hist, edges):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45,60)
        ax.plot(X, Y, Z, 'k.', alpha=0.3)
        X, Y = np.meshgrid(edges[0][:-1], edges[1][:-1])
        for ct in [0,2,5,7,9]:
            cs = ax.contour(X,Y,hist[:,:,ct], zdir = 'z', offset=edges[2][ct], level=100,cmap=plt.cm.RdYlBu_r, alpha=0.5)
        plt.colorbar(cs)
        plt.show()


    def getLabels(self, key):
        labels= {}
        labels['cartesian'] = ['x', 'y', 'z']
        labels['rotation'] = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']
        labels['linearVelocity'] = ["x'", "y'", "z'"]
        labels['angularVelocity'] = ["alpha","beta'","gamma'"]
        labels['grasperAngle'] = ["theta"]
        return labels[key]


    def orientation(self,rotationMatrix):
        #print ("size of the rotationMatrix {}".format(rotationMatrix.shape))
        _ori = np.zeros((len(rotationMatrix),6))
        for i in range(_ori.shape[0]):
            for j in range(_ori.shape[1]):
                _ori[i][0] = math.atan2(rotationMatrix[i][7], rotationMatrix[i][8])
                _ori[i][1] = math.atan2(-rotationMatrix[i][6], (rotationMatrix[i][8]**2 +rotationMatrix[i][7]**2)**0.5)
                _ori[i][2] = math.atan2(rotationMatrix[i][3], rotationMatrix[i][0])
                _ori[i][3] = math.atan2(rotationMatrix[i][16], rotationMatrix[i][17])
                _ori[i][4] = math.atan2(-rotationMatrix[i][15],(rotationMatrix[i][17]**2 + rotationMatrix[i][16]**2)**0.5)
                _ori[i][5] = math.atan2(rotationMatrix[i][12], rotationMatrix[i][9])
        return _ori

path = os.path.abspath(os.path.dirname(sys.argv[0]))
npds = needlePassing(path)
