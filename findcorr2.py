import os, sys, glob, math
import numpy as np
import pandas as pd
from sklearn import externals
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal
class fcr:
    def __init__(self, _dir):
        self._dir =_dir

    def findCrossCorrelation(self, subject):
        corrVariables = ["/cartesian","/rotation"]
        kinematicsPath = self._dir + "/segments/" + subject + "/*"
        covarPath = self._dir + "/covariances/"
        covarList = []
        if not os.path.exists(covarPath):
            os.makedirs(covarPath)
        for variable in glob.glob(kinematicsPath):
            kin =  (variable.split("/")[len(variable.split("/"))-1])
            covarList = []
            for segment in glob.glob(variable+"/*"):
                segmentName = segment.split("/")[len(segment.split("/"))-1]
                kinematics = externals.joblib.load(segment)
                left_covariance = np.cov(kinematics[:,0:kinematics.shape[1]/2], rowvar = False).reshape(-1,1)
                right_covariance = np.cov(kinematics[:,kinematics.shape[1]/2:kinematics.shape[1]], rowvar = False).reshape(-1,1)
                covarList.append([segmentName, np.split(left_covariance, len(left_covariance)), np.split(right_covariance,len(left_covariance))])
            covarList = pd.DataFrame(data  = covarList, columns = ["gestures", "left_covariance", "right_covariance"])
            covarList.to_csv("{}/{}{}.csv".format(covarPath,subject, kin))
    def collateCovariance(self):
        """
        this function iterates over all files of a given variable and collates covariance of a specific segment, using dicts
        """
        segmentdict = {}
        covarPath = self._dir + "/covariances/*cartesian.csv"
        for name in glob.glob(covarPath):
            segments = pd.read_csv(name)[["gestures"]]
            for gesture in segments:
                segmentdict[gesture] = []
        print segmentdict

    def gatherDistribution(self, experts, skill):
        kinematicsPath = self._dir + "/segments/" + experts[0] + "/*"
        covarPath = self._dir + "/covariances/"
        if not os.path.exists(covarPath):
            os.makedirs(covarPath)
        for variable in glob.glob(kinematicsPath):
            kin =  (variable.split("/")[len(variable.split("/"))-1])
            covarList = {}
            for segment in glob.glob(variable+"/*"):
                distrbution = []
                segmentName = segment.split("/")[len(segment.split("/"))-1]
                distrbution = externals.joblib.load(segment)
                for i in range(1,len(experts)):
                    segment = segment.replace(experts[0], experts[i])
                    #print ("segmentName {} kinematics {} subject {}".format(segmentName, kin, experts[i]))
                    if os.path.exists(segment):
                        distrbution = np.concatenate((distrbution, externals.joblib.load(segment)))
                #print ("distrbution shape {}".format(distrbution.shape))
                left_covariance = np.cov(distrbution[:,0:distrbution.shape[1]/2], rowvar = False).reshape(-1,1)
                left_temp = []
                for left in left_covariance:
                    left_temp.append(float(left[0]))
                right_temp = []
                right_covariance = np.cov(distrbution[:,distrbution.shape[1]/2:distrbution.shape[1]], rowvar = False).reshape(-1,1)
                for right in right_covariance:
                    right_temp.append(float(right[0]))
                covarList[segmentName] =  [np.asarray(left_temp), np.asarray(right_temp)]
                externals.joblib.dump(covarList, "{}/{}{}.p".format(covarPath, skill, kin))


    def compareCovariance(self):
        covarPath = self._dir + "/covariances/csvs/"
        kin = "grasperAngle"
        skill = "intermediary"
        ExpertCovDictLeft = externals.joblib.load("/home/uva-dsa1/Downloads/dVRKvideos/Needle_Passing/covariances/experts{}.p".format(kin))
        NoviceCovDictLeft = externals.joblib.load("/home/uva-dsa1/Downloads/dVRKvideos/Needle_Passing/covariances/{}{}.p".format(skill,kin))
        left_covariance = {}
        right_covariance = {}
        for key in ExpertCovDictLeft:
            if key in NoviceCovDictLeft.keys():
                left_covariance[key] =  ExpertCovDictLeft[key][1]-NoviceCovDictLeft[key][1]
                right_covariance[key] =  ExpertCovDictLeft[key][1]-NoviceCovDictLeft[key][1]
        _left = pd.DataFrame.from_dict(left_covariance)
        _right = pd.DataFrame.from_dict(right_covariance)
        _left.to_csv("{}/Left{}{}.csv".format(covarPath,skill, kin))
        _right.to_csv("{}/Right{}{}.csv".format(covarPath,skill, kin))
_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
fcr = fcr(_dir)
#mpvds.loopSubjects()

experts = ['Needle_Passing_F001', 'Needle_Passing_F004', 'Needle_Passing_H005', 'Needle_Passing_I005', 'Needle_Passing_H004']
subjects = ['Needle_Passing_B003', 'Needle_Passing_B004', 'Needle_Passing_C001', 'Needle_Passing_C004', 'Needle_Passing_C005', 'Needle_Passing_D002', 'Needle_Passing_D003', 'Needle_Passing_D004', 'Needle_Passing_D005', 'Needle_Passing_E001','Needle_Passing_E003', 'Needle_Passing_E004', 'Needle_Passing_E005',  'Needle_Passing_F003', 'Needle_Passing_H002',  'Needle_Passing_I002', 'Needle_Passing_I003','Needle_Passing_I004']
#subjects = ['Needle_Passing_C004']
novices = ['Needle_Passing_B001', 'Needle_Passing_B002','Needle_Passing_C002', 'Needle_Passing_C003','Needle_Passing_D001',]
"""
fcr.gatherDistribution(subjects, "intermediary")
fcr.gatherDistribution(experts, "experts")
fcr.gatherDistribution(novices, "novices")
"""
fcr.compareCovariance()
"""
for subject in subjects:
    fcr.findCrossCorrelation(subject)
fcr.collateCovariance()
"""
