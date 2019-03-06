import os, sys, glob, math
import numpy as np
import pandas as pd
from sklearn import externals
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fastdtw import fastdtw
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
#                    _expertDemonstration = self.flattenArray(_expertDemonstration, gesture.shape[1])
                    comparePath1 = comparePath + "/{}/{}".format(kin,gestureName)
                    if not os.path.exists(comparePath1):
                        os.makedirs(comparePath1)
#                    _list.append([gestureName, self.plotHistogram(_expertDemonstration, gesture, labelKeys, gesture.shape[1]/2, comparePath1, "experts", "red", gestureName)])
                    _list.append([gestureName, self.plotTrajectory(_expertDemonstration, gesture, labelKeys, gesture.shape[1]/2, comparePath1, "experts", "red", gestureName)])
            df = pd.DataFrame(data  = (sorted(_list, key = self.sortSecond)), columns = ["gestures", "intersection"])
            df.to_csv("{}/{}{}.csv".format(comparePath,subject, kin))

    def plotHistogram(self, trajectory, subject, labelKeys, subplots, segmentPath, performance, _color, gestureName):
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
                ax.set_ylabel('PDF {}'.format(labelKeys[i%subplots]), fontsize = 8)
#                ax.set_title('Distribution for {} for {}'.format(labelKeys[i%subplots], manip), fontsize  =5)
                ax.text(23, 45, r'$\mu=15, b=3$')
                n, bins, patches = ax.hist(x=trajectory[:,i], bins=100, color=_color, alpha=0.7, rwidth=1.1, weights = np.ones_like(trajectory[:,i])/len(trajectory[:,i]))
                n, bins, patches = ax.hist(x=subject[:,i], bins=100, color="white", alpha=0.7, rwidth=1.1, weights = np.ones_like(subject[:,i])/len(subject[:,i]))
                hist_1, _ = np.histogram(trajectory[:,i], bins=100, density = True)
                hist_2, _ = np.histogram(subject[:,i], bins=100, density = True)
                intersections.append(self.findIntersection(hist_1, hist_2, bins))
            red_patch = mpatches.Patch(color = 'darkred', label = 'expert')
            blue_patch = mpatches.Patch(color = 'white', label = 'novice')
            plt.legend(handles= [red_patch, blue_patch])
            plt.savefig("{}/{}{}.pdf".format(segmentPath, gestureName, manip), dpi = 100, bbox = 'tight')                                  
            plt.close()
        _meanIntersection = self.findMean(np.array(intersections).reshape(-1,subplots*2))
        return _meanIntersection

    def dtwTrajectories(self, experts, subject):
        print ("subject: {}".format(subject))
        expertkinematicsPath = self._dir + "/segments/" + experts[0] + "/*"
        subjectkinematicsPath = self._dir + "/segments/" + subject 
        comparePath = self._dir + "/figures/comparision/" + subject
        for name in glob.glob(expertkinematicsPath):
            kin =  (name.split("/")[len(name.split("/"))-1])
            _list = []
            for num in glob.glob(name+ "/*"):                
                gestureName =  (num.split("/")[len(num.split("/"))-1].replace(".p",""))
                expertGesturePath =  num
                _expertDemonstrations = []
                _expertDemonstrations.append(externals.joblib.load(expertGesturePath)) 
                for i in range(1, len(experts)):
                    expertGesturePath1 = expertGesturePath.replace(experts[0],experts[i])
                    if os.path.exists(expertGesturePath1):
                        _expertDemonstrations.append(externals.joblib.load(expertGesturePath1))
                subjectGesturePath = expertGesturePath.replace(experts[0], subject)
                if os.path.exists(subjectGesturePath):
                    _subjectDemonstration = externals.joblib.load(subjectGesturePath)
                    labelKeys = self.getLabels(kin)
                    _list.append([ gestureName, self.compareTrajectories(_expertDemonstrations, _subjectDemonstration)])                
                if kin == "cartesian":
                    self.plotmultivariateDistributions(_expertDemonstrations,_subjectDemonstration, subject, gestureName)
            #print (sorted(_list, key = self.sortSecond)) 
            df = pd.DataFrame(data  = (sorted(_list, key = self.sortSecond)), columns = ["gestures", "intersection"])
            df.to_csv("{}/{}{}dtw.csv".format(comparePath,subject, kin))

    def compareplots(self, experts, subject):    
        expertkinematicsPath = self._dir + "/segments/" + experts[0] + "/*"
        subjectkinematicsPath = self._dir + "/segments/" + subject 
        histogramPath = self._dir + "/figures/histogram/"
        comparePath = self._dir + "/figures/comparision/" + subject
        for name in glob.glob(expertkinematicsPath):
            kin =  (name.split("/")[len(name.split("/"))-1])
            _list = []
            for num in glob.glob(name+ "/*"):                
                gestureName =  (num.split("/")[len(num.split("/"))-1].replace(".p",""))
                expertGesturePath =  num
                _expertDemonstrations = []
                _expertDemonstrations.append(externals.joblib.load(expertGesturePath)) 
                for i in range(1, len(experts)):
                    expertGesturePath1 = expertGesturePath.replace(experts[0],experts[i])
                    if os.path.exists(expertGesturePath1):
                        _expertDemonstrations.append(externals.joblib.load(expertGesturePath1))
                subjectGesturePath = expertGesturePath.replace(experts[0], subject)
                if os.path.exists(subjectGesturePath):
                    _subjectDemonstration = externals.joblib.load(subjectGesturePath)
                    labelKeys = self.getLabels(kin)
                    comparePath1 = comparePath + "/{}/{}".format(kin,gestureName)
                    if not os.path.exists(comparePath1):
                        print ("loading expert trajectories") 
                        os.makedirs(comparePath1)
                    _list.append([self.compareTrajectories(_expertDemonstrations, _subjectDemonstration), labelKeys])                

    def plotTrajectory(self, experts, subject, labelKeys, subplots, segmentPath, performance, _color, gestureName):
        manip = "left"
        print ("loading expert trajectories") 
        subplotnum1 = int("{}1".format(subplots))
        for k in range(2):
            fig = plt.figure()
            if k>0:
                manip = "right"
            for i in range(0+k*subplots, (k+1)*subplots):
                plotNum = int("{}{}".format(subplotnum1, (i%subplots+1)))
                ax = fig.add_subplot(plotNum)
                ax.grid(axis='y', alpha=0.75)
                ax.set_xlabel('Time', fontsize = 8)
                ax.set_ylabel('Value of {}'.format(labelKeys[i%subplots]), fontsize = 8)
#                ax.set_title('Distribution for {} for {}'.format(labelKeys[i%subplots], manip), fontsize  =5)
                ax.text(23, 45, r'$\mu=15, b=3$')
                for trajectory in experts:
                    ax.plot(trajectory[:,i], color = 'red')
                ax.plot(subject[:,i], color = 'green')
            red_patch = mpatches.Patch(color = 'red', label = 'expert')
            blue_patch = mpatches.Patch(color = 'green', label = 'novice')
            plt.legend(handles= [red_patch, blue_patch])
            plt.savefig("{}/kin{}{}.pdf".format(segmentPath, manip,gestureName), dpi = 100, bbox = 'tight')                                 
        plt.close() 
        return

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
         labels['rotation'] = ['roll', 'pitch', 'yaw']
         labels['linearVelocity'] = ["x'", "y'", "z'"]
         labels['angularVelocity'] = ["alpha","beta'","gamma'"]
         labels['grasperAngle'] = ["theta"]
         return labels[key]

    def findIntersection(self, hist1, hist2, bins):
        bins  = np.diff(bins)
        intersection = 0
        for i in range(len(bins)):
            intersection += min(bins[i]*hist1[i], bins[i]*hist2[i])
#        minima = np.minimum(hist1, hist2)
#        intersection = np.true_divide(np.sum(minima), np.sum(hist2))
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

    def compareTrajectories(self, expertsTraj, subjectTraj):
        """ 
        This function compares the trajectories based on the fast dtw
        """
        #print (len(expertdemonstrations))
        dev = []
        for expertTraj in expertsTraj:
            distance, path = fastdtw(expertTraj, subjectTraj)
            dev.append(distance)
        #print dev
        return min(dev)/float(len(subjectTraj))

    def plotmultivariateDistributions(self, experts, subject, subjectName, gestureName):
        print ("Subject shape expert shaper {}  {}".format(subject.shape, len(experts)))        
        comparePath = self._dir + "/figures/comparision/" + subjectName
        figLeft = plt.figure()
        figRight = plt.figure()
        for expert in experts:
            r, phi, theta = self.multivariateDistributions(expert)
            plotNum = 311
            ax = figLeft.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Euclidean', fontsize = 8)
            ax.plot(r[:,0], color = 'red')
            plotNum = 312
            ax = figLeft.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Phi', fontsize = 8)
            ax.plot(phi[:,0], color = 'red')
            plotNum = 313
            ax = figLeft.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Theta', fontsize = 8)
            ax.plot(theta[:,0], color = 'red')

            plotNum = 311
            ax = figRight.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Euclidean', fontsize = 8)
            ax.plot(r[:,1], color = 'red')
            plotNum = 312
            ax = figRight.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Phi', fontsize = 8)
            ax.plot(phi[:,1], color = 'red')
            plotNum = 313
            ax = figRight.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Theta', fontsize = 8)
            ax.plot(theta[:,1], color = 'red')
        
        for sub in subject:
            r, phi, theta = self.multivariateDistributions(subject)
            plotNum = 311
            ax = figLeft.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Euclidean', fontsize = 8)
            ax.plot(r[:,0], color = 'green')
            plotNum = 312
            ax = figLeft.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Phi', fontsize = 8)
            ax.plot(phi[:,0], color = 'green')
            plotNum = 313
            ax = figLeft.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Theta', fontsize = 8)
            ax.plot(theta[:,0], color = 'green')

            plotNum = 311
            ax = figRight.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Euclidean', fontsize = 8)
            ax.plot(r[:,1], color = 'green')
            plotNum = 312
            ax = figRight.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Phi', fontsize = 8)
            ax.plot(phi[:,1], color = 'green')
            plotNum = 313
            ax = figRight.add_subplot(plotNum)
            ax.set_xlabel('Time', fontsize = 8)
            ax.set_ylabel('Theta', fontsize = 8)
            ax.plot(theta[:,1], color = 'green')
        print ("comparePath {}".format(comparePath))
        figLeft.savefig("{}/multivar/multivarLeft{}.pdf".format(comparePath, gestureName), dpi = 100, bbox = 'tight')                                 
        figRight.savefig("{}/multivar/multivarRight{}.pdf".format(comparePath, gestureName), dpi = 100, bbox = 'tight')                                 
        plt.close() 

    def multivariateDistributions(self, cartesians):
        """
        This function takes converts the x-y-z values to polar co-ordinates
        """
        #print ("size of cartesians {}".format(cartesians.shape))
        r = np.zeros((cartesians.shape[0],2))    
        phi = np.zeros((cartesians.shape[0],2))
        theta = phi.copy()    
        r[:,0] = cartesians[:,0]**2 +cartesians[:,1]**2 + cartesians[:,2]**2
        r[:,0] = np.array(r[:,0]**0.5)
        r[:,1] = cartesians[:,3]**2 +cartesians[:,4]**2 + cartesians[:,5]**2
        r[:,1] = np.array(r[:,1]**0.5)
        for i in range(phi.shape[0]):
            phi[i,0] = math.atan2(cartesians[i,1],cartesians[i,0])
            theta[i,0] = math.acos(cartesians[i,2]/r[i,0])
            phi[i,1] = math.atan2(cartesians[i,4],cartesians[i,3])
            theta[i,1] = math.acos(cartesians[i,5]/r[i,1])
        return r, phi, theta 
_dir = os.path.abspath(os.path.dirname(sys.argv[0]))    
mpvds = mapVids(_dir)
#mpvds.loopSubjects()
experts = ['Needle_Passing_F001', 'Needle_Passing_F004', 'Needle_Passing_H005', 'Needle_Passing_I005', 'Needle_Passing_H004']
subjects = ['Needle_Passing_B001']
#, 'Needle_Passing_B002', 'Needle_Passing_B003', 'Needle_Passing_B004', 'Needle_Passing_C001','Needle_Passing_C002', 'Needle_Passing_C003', 'Needle_Passing_C004', 'Needle_Passing_C005', 'Needle_Passing_D001','Needle_Passing_D002', 'Needle_Passing_D003', 'Needle_Passing_D004', 'Needle_Passing_D005', 'Needle_Passing_E001','Needle_Passing_E003', 'Needle_Passing_E004', 'Needle_Passing_E005', 'Needle_Passing_F001', 'Needle_Passing_F003','Needle_Passing_F004', 'Needle_Passing_H002', 'Needle_Passing_H004', 'Needle_Passing_H005', 'Needle_Passing_I002', 'Needle_Passing_I003','Needle_Passing_I004', 'Needle_Passing_I005' ]
#subjects = ['Needle_Passing_C004']

for subject in subjects:
    mpvds.dtwTrajectories(experts, subject)    
