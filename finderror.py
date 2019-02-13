import os, sys, glob
import numpy as np
import pandas as pd
from sklearn import externals
from modelerror import needlePassing

class findError:
    def __init__(self, path):
        self.dataPath = path
        self.loadDemonstrations("/kinematics")        

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
    def loadDemonstrations(self, key):
        """
        This function globs over all the demonstrations for the given key (kinematics, transcriptions, video) and calls the plotgraph function after completing globbing  
        """
        demonstrationsPath = self.dataPath + key
        globPath = demonstrationsPath + "/**/*"
        kinOffset, kinSpan = self.loadOffsets()
        for name in glob.glob(globPath):
            if key == "/kinematics":
                for key1, value in kinOffset.iteritems() :
                    constraints = self.loadConstraints(key1)
                    cartesians = self.readCartesians(name, kinOffset[key1], kinSpan[key1])                
                    transcriptFile = name.replace(key, "/transcriptions").replace("AllGestures", "")
                    transcript = self.readTranscripts(transcriptFile)                
                    segmentPath = (transcriptFile.replace("/transcriptions", "/segments").replace(".txt","/")) + str(key1)
                    if not transcript == []:
                        segmentMap = self.makeSegments(cartesians, transcript)
                        self.checkConstraints(cartesians, constraints, segmentMap)
                    
            else:
                print name
               #self.readVideo(name)

    def readCartesians(self, demonstration, offset, span):
        """ 
        This function reads the cartesian values from the kinematics file for each demonstration
        """
        df = np.array(pd.read_csv(demonstration, delimiter = '    ', header = None))
        psm_offset = df.shape[1]/2
        cartesians = np.concatenate((df[:,psm_offset+offset:psm_offset+offset+span], df[:, psm_offset+psm_offset/2 + offset:psm_offset+psm_offset/2+offset + span]), axis=1) 
        return cartesians

    def loadConstraints(self, key):
        constraintsPath = self.dataPath + "/constraints/"
        pickleFile = constraintsPath + str(key) + ".p"
        constraints = externals.joblib.load(pickleFile)
        print ("constraint loaded for {}".format(key))
        return constraints

    def checkConstraints(self, cartesians, constraints, segmentMap):
        #print ("checking constraints using file {} ".format(constraints))
        count = 0
#        print ("length {}".format(len(constraints)))
        for key, value in segmentMap.iteritems():
            try:
 #               print ("checking constraints for key {}".format(key))
                localConstraint = constraints[key]
                if len(localConstraint)>0:                
                    segment = cartesians[value[0]:value[1], 0:3]                 
                    self.findViolations(segment, localConstraint, value)
                
            except KeyError:
                count +=1
                print ("no constraints loaded for segment {} {}".format(key, count))

    def loadMetaFile(self):
        """
        This function loads the meta_file for needle_passing which gives the category-wise score for each demonstration along with the total score
        """
        scores = {}
        metaFilePath = self.dataPath + "/meta_file_Needle_Passing.txt"
        for name in glob.glob(metaFilePath):
            df = np.array(pd.read_csv(name, delimiter='\t', engine='python', header=None))
            for i in range(df.shape[0]):
                scores[df[i][0]] = []
                scores[df[i][0]].append(df[i,1:])
            
            return scores

    def readTranscripts(self, transcript):
        """
        This function reads the transcript file for each demonstration
        """
        try:
            df = np.array(pd.read_csv(transcript, delimiter=' ', header=None))
            return df
        except IOError:
            return []

    def makeSegments(self, cartesians, transcript):
        """
        This function segments the trajectory based on the boundaries obtained from the transcript file
        """
        uniqueGestures =  set(transcript[:,2])
        tree = list(transcript[:,2])
        segmentDict = {}
        countSegment = {}
        segment2packet = {}
        for gesture in uniqueGestures:
            countSegment[gesture] = 0
        for i in range(transcript.shape[0]):
            segment = cartesians[transcript[i][0]:transcript[i][1]]
            prev = transcript[i][2]
            countSegment[transcript[i][2]] +=1
            #print (segmentDict[transcript[i][2]])
            seg = "{}->{}.p".format(transcript[i][2], countSegment[transcript[i][2]])
            segment2packet[seg] = [transcript[i][0], transcript[i][1]]
        return segment2packet

    def findViolations(self, segment, constraint,_range):
        assert (segment.shape[0] == _range[1] - _range[0]), "sanity check! {} {}".format(segment.shape[0], _range[1]-_range[0])
        constraint = np.array(constraint).reshape(-1,1)
        print ("constraint {}".format(constraint.shape))
        print ((i for i,v in enumerate(segment) if v[0] > constraint[0][0]))


#        print ("Checking cartesians for {} with constrain {}".format(_range, constraint))

_path = os.path.abspath(os.path.dirname(sys.argv[0]))
findError = findError(_path)
