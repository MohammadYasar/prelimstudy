import os, sys, glob, math
import numpy as np
import pandas as pd
from sklearn import externals
from modelerror import needlePassing
from sparse_filtering import SparseFiltering
import matplotlib.pyplot as plt
from hmmlearn import hmm

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
        failure_dict = self.mapdemonstrationtofailures()
        key = "diagnose"
        for name in glob.glob(globPath):
            if key == "/kinematics":
                for key1, value in kinOffset.iteritems() :
                    #constraints = self.loadConstraints(key1)
                    cartesians = self.readCartesians(name, kinOffset[key1], kinSpan[key1])
                    transcriptFile = name.replace(key, "/transcriptions").replace("AllGestures", "")
                    transcript = self.readTranscripts(transcriptFile)
                    if not transcript == []:
                        self.mapPacketstoSegments(name, cartesians, transcript, failure_dict)
                    segmentPath = (transcriptFile.replace("/transcriptions", "/segments").replace(".txt","/")) + str(key1)
                    """
                    if not transcript == []:
                        segmentMap = self.makeSegments(cartesians, transcript)
                        self.checkConstraints(cartesians, constraints, segmentMap)
                    """
            elif key == "diagnose":
                errorArray = self.findImportantFeatures(name)
                """
                if not errorArray == []:
                    externals.joblib.dump(errorArray, "errorArray.p")
                    self.filterFeatures(errorArray)
                """
            else:
                print name
               #self.readVideo(name)

    def filterFeatures(self, dataset):
        data = dataset[:,0:38]
        n_features = 20
        estimator = SparseFiltering(n_features=n_features, maxfun=500,  iprint=50)
        features = estimator.fit_transform(data)
        """
        plt.hist(features.flat, bins=50)
        plt.xlabel("Activation")
        plt.ylabel("Count")
        _ = plt.title("Feature activation histogram")
        """
        print ("features shape {}".format(features.shape))
        print ("features {}".format(features))
        activated_features = (features > 0.1).mean(0)
        plt.hist(activated_features)
        plt.xlabel("Feature activation ratio over all examples")
        plt.ylabel("Count")
        _ = plt.title("Lifetime Sparsity Histogram")
        plt.show()

    def modelHMM(self, data):
        #either use a generative approach or a discriminative one
        model = hmm.GaussianHMM(n_components=2, covariance+type="full", n_iter=100)
        model.fit(data)

    def mapdemonstrationtofailures(self):
        failure_dict = {}
        _filename = self.dataPath + "/failure_times.csv"
        _df = pd.read_csv(_filename)
        for index, row  in _df.iterrows():
            if not math.isnan(_df.at[index, 'S1']):
                failure_dict[_df.at[index, 'Demonstration']] = [int(_df.at[index, 'S1']),int(_df.at[index, 'E1'])]

        return failure_dict

    def mapPacketstoSegments(self, name, kinematics, transcript, failure_dict):
        error_transcript = np.zeros((kinematics.shape[0],3))
        for i in range(error_transcript.shape[0]):
            error_transcript[i][0]=i

        failure_time1 = 0; failure_time2 = 0
        subject = name.split("/")[len(name.split("/"))-1].replace(".txt","")
        if subject in failure_dict.keys():
            failure_time1 = failure_dict[subject][0]*30
            failure_time2 = failure_dict[subject][1]*30

        for i in range(transcript.shape[0]):
            print ("t1 {} t2 {} g{}".format(transcript[i][0],transcript[i][1], int(transcript[i][2].replace("G",""))))
            error_transcript[int(transcript[i][0]):int(transcript[i][1]),1] = int(transcript[i][2].replace("G",""))
            if failure_time1 in range(transcript[i][0], transcript[i][1]):
                ("segment where failure started {}".format(transcript[i][2]))
            if failure_time2 in range(transcript[i][0], transcript[i][1]):
                ("segment where failure ended {}".format(transcript[i][2]))
        error_transcript[failure_time1:failure_time2,2] = 1
        print (error_transcript)
        error_df = pd.DataFrame(data = error_transcript, columns = ['packet', 'segment', 'error'])
        error_df.to_csv(self.dataPath + "/transcriptions/error{}.csv".format(subject))

    def findImportantFeatures(self, subject):
        df = np.asarray(pd.read_csv(subject, delimiter = '    ', header = None))
        psm_offset = df.shape[1]/2
        cartesians = df[:,psm_offset:]
        errorArray = []
        subject = subject.split("/")[len(subject.split("/"))-1].replace(".txt","")
        errorTranscript = self.dataPath + "/transcriptions/error{}.csv".format(subject)
        if os.path.exists(errorTranscript):
            error = np.asarray(pd.read_csv(errorTranscript))
            for a in range(0, len(error)):
                if error[a][2]==11 and error[a][3]==1:
                    print ("error found in 5 for {} in subject {}".format(a, subject))

            errorArray = np.concatenate((cartesians, error[:,2:]), axis=1)
            errordf = pd.DataFrame(errorArray)
            errordf.to_csv(self.dataPath + "/suboptimals/error{}.csv".format(subject))
        return errorArray

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


_path = os.path.abspath(os.path.dirname(sys.argv[0]))
findError = findError(_path)
