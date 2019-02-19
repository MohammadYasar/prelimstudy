import os, sys, glob
import numpy as np
import pandas as pd
from sklearn import externals
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class mapVids:
    def __init__(self, _dir):
        self._dir =_dir
        
#    def load
    def compareSub(self, subject):    
        kinematicsPath = self._dir + "/segments/" + subject + "/*"
        histogramPath = self._dir + "/figures/histogram/"
        comparePath = self._dir + "/figures/comparision/" + subject
        for name in glob.glob(kinematicsPath):
            kin =  (name.split("/")[len(name.split("/"))-1])
            _list = []
            for num in glob.glob(name+ "/*"):                
                gestureName =  (num.split("/")[len(num.split("/"))-1].replace(".p",""))
                gesture = externals.joblib.load(num)
                _expertPath = histogramPath + kin + "/" + gestureName + ".p/experts.p"
                if os.path.exists(_expertPath):
                    _expertDemonstration = externals.joblib.load(_expertPath) 
                    labelKeys = self.getLabels(kin)
                    _expertDemonstration = self.flattenArray(_expertDemonstration, gesture.shape[1])
                    comparePath1 = comparePath + "/{}/{}".format(kin,gestureName)
                    if not os.path.exists(comparePath1):
                        os.makedirs(comparePath1)
                    _list.append([gestureName, self.plotHistogram(_expertDemonstration, gesture, labelKeys, gesture.shape[1]/2, comparePath1, "experts", "red")])
            print (sorted(_list, key = self.sortSecond)) 
            df = pd.DataFrame(data  = (sorted(_list, key = self.sortSecond)), columns = ["gestures", "intersection"])
            df.to_csv("{}/{}{}.csv".format(comparePath,subject, kin))

    def plotHistogram(self, trajectory, subject, labelKeys, subplots, segmentPath, performance, _color):
        manip = "left"
        plt.style.use('dark_background')
        subplotnum1 =int("{}1".format(subplots))
        intersections = []
        for k in range(2):
            fig = plt.figure()
            if k>0:
                manip = "right"
            for i in range(0+k*subplots,(k+1)*subplots):
                plotNum = int("{}{}".format(subplotnum1, (i%subplots+1)))
                ax = fig.add_subplot(plotNum)
                ax.grid(axis='y', alpha=0.75)
                ax.set_xlabel('Values', fontsize = 8)
                ax.set_ylabel('Frequency of {}'.format(labelKeys[i%subplots]), fontsize = 8)
#                ax.set_title('Distribution for {} for {}'.format(labelKeys[i%subplots], manip), fontsize  =5)
                ax.text(23, 45, r'$\mu=15, b=3$')
                n, bins, patches = ax.hist(x=trajectory[:,i], bins=100, color=_color, alpha=0.7, rwidth=1.1, density = True)
                n, bins, patches = ax.hist(x=subject[:,i], bins=100, color="white", alpha=0.7, rwidth=1.1, density = True)
                hist_1, _ = np.histogram(trajectory[:,i], bins=100)
                hist_2, _ = np.histogram(subject[:,i], bins=100)
                intersections.append(self.findIntersection(hist_1, hist_2))
            red_patch = mpatches.Patch(color = 'darkred', label = 'expert')
            blue_patch = mpatches.Patch(color = 'white', label = 'novice')
            plt.legend(handles= [red_patch, blue_patch])
            plt.savefig("{}/{}.pdf".format(segmentPath, manip), dpi = 100, bbox = 'tight')                                  
        plt.close()
        _meanIntersection = self.findMean(np.array(intersections).reshape(-1,subplots*2))
        return _meanIntersection

    def flattenArray(self, trajectory, col):
        row = 0
        for traj in trajectory:
            row+= len(traj)
        trajTensor = np.zeros((row,col))
        _iter = 0
        for traj in trajectory:
            trajTensor[_iter:_iter+len(traj)] = traj[:,:]
            _iter += len(traj)
        return trajTensor

    def getLabels(self, key):
         labels= {}
         labels['cartesian'] = ['x', 'y', 'z']
         labels['rotation'] = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']
         labels['linearVelocity'] = ["x'", "y'", "z'"]
         labels['angularVelocity'] = ["alpha","beta'","gamma'"]
         labels['grasperAngle'] = ["theta"]
         return labels[key]

    def findIntersection(self, hist1, hist2):
        minima = np.minimum(hist1, hist2)
        intersection = np.true_divide(np.sum(minima), np.sum(hist2))
        return intersection
    
    def findMean(self,_list):
        _array = np.array(_list)
#        for i in range(_mean.shape[0]):
        _mean = np.mean(_array[:,int(_array.shape[1]/2):_array.shape[1]], axis=1)
        return _mean[0]

    def sortSecond(self,val): 
        return val[1]  
    
    def loopSubjects(self):
        subjects = self._dir + "/transcriptions/*"
        for name in glob.glob(subjects):
            print (name) 
            subject = name.split("/")[len(name.split("/"))-1].replace(".txt","")
            self.compareSub(subject)
_dir = os.path.abspath(os.path.dirname(sys.argv[0]))    
mpvds = mapVids(_dir)
mpvds.loopSubjects()
