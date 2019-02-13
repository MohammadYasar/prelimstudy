import sys, os, glob, cv2, math
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn import externals

class needlePassing:
    def __init__(self, dataPath):
        self.dataPath = dataPath
#        self.loadDemonstrations("/kinematics")
        self.plotAllInformation()

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
        scores = self.loadMetaFile()
        globPath = demonstrationsPath + "/**/*"
        color = ""
        novices = []
        intermediary = []
        experts = []
        novices_scores = []
        intermediary_scores = []
        expert_scores = []
        kinOffset, kinSpan = self.loadOffsets()
        for name in glob.glob(globPath):
            if key == "/kinematics":
                for key1, value in kinOffset.iteritems() :
                    print ("reading for {} with offset {} and span {}".format(key1, kinOffset[key1], kinSpan[key1]))
                    cartesians = self.readCartesians(name, kinOffset[key1], kinSpan[key1])                
                    transcriptFile = name.replace(key, "/transcriptions").replace("AllGestures", "")
                    transcript = self.readTranscripts(transcriptFile)                
                    segmentPath = (transcriptFile.replace("/transcriptions", "/segments").replace(".txt","/")) + str(key1)
                    print ("segmentPath %s"%segmentPath)
                    if not transcript == []:
                        if not os.path.exists(segmentPath):
                            os.makedirs(segmentPath)
                        self.makeSegments(cartesians, transcript, segmentPath)
                    subject = (name.replace(demonstrationsPath,"").replace("AllGestures", "").replace(".txt", "").replace("/", ""))                
                    
            else:
                print name
               #self.readVideo(name)

    def readTrajectoryScores(self, scores, novices, experts,intermediary, cartesians, subject, expert_scores, intermediary_scores, novices_scores):
        """
        This function reads the scores from the metafile dataframe and color codes the demonstrations according to the score range
        """
        for i, value in enumerate(scores):
            if (scores[i][0] == subject):
                    if scores[i][3] <10 :
                        color = "red"
                        novices.append(cartesians)
                        novices_scores.append(scores[i][3])
                    elif scores[i][3]<20:
                        color = "yellow"
                        intermediary.append(cartesians)
                        intermediary_scores.append(scores[i][3])
                    else:
                        color = "blue"
                        experts.append(cartesians)
                        expert_scores.append(scores[i][3])            
    
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

    def readCartesians(self, demonstration, offset, span):
        """ 
        This function reads the cartesian values from the kinematics file for each demonstration
        """
        df = np.array(pd.read_csv(demonstration, delimiter = '    ', header = None))
        psm_offset = df.shape[1]/2
        cartesians = np.concatenate((df[:,psm_offset+offset:psm_offset+offset+span], df[:, psm_offset+psm_offset/2 + offset:psm_offset+psm_offset/2+offset + span]), axis=1) 
        return cartesians

    def readVideo(self, demonstration):
        """
        This file reads the frames from video file for each demonstration and applies density based optical flow
        """
        cap = cv2.VideoCapture(demonstration)
        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        while(1):
            ret, frame2 = cap.read()
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            next = cv2.bilateralFilter(next,9,75,75)
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2',bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png',frame2)
                cv2.imwrite('opticalhsv.png',bgr)
            prvs = next
        cap.release()
        cv2.destroyAllWindows()

    def compareTrajectories(self, expertdemonstrations, novicedemonstrations, expert_scores, novice_scores):
        """ 
        This function compares the trajectories based on the fast dtw
        """
        #print (len(expertdemonstrations))
        for i in range(len(novicedemonstrations)):
            distance, path = fastdtw(expertdemonstrations[0], novicedemonstrations[i])
            print "{} {} {}".format(novice_scores[i],expert_scores[1], distance)

    def readTranscripts(self, transcript):
        """
        This function reads the transcript file for each demonstration
        """
        try:
            df = np.array(pd.read_csv(transcript, delimiter=' ', header=None))
            return df
        except IOError:
            pass
        return []

    def plotAllInformation(self):
        kinOffset, kinSpan = self.loadOffsets()
        scores = self.loadMetaFile()
        key1 = "/segments/" 
        for key, value in kinOffset.iteritems():
            uniqueSegments, segmentPaths = self.gatherSegments(key1, key)
            self.mapSegmentstoScores(uniqueSegments, segmentPaths, scores, str(key), kinSpan[key])

    def plotGraph(self, cartesians, scores, gesture, expertise, datatype):  
        imagePath = self.dataPath + "/figures/"
        segmentPath = imagePath + "%s/"%gesture + datatype
        if not os.path.exists(segmentPath):
            os.makedirs(segmentPath)
        fig = plt.figure()
        fig.set_size_inches(30,20)
        subplot_num1 = int("{}1".format(len(cartesians)))
        for i, value in enumerate(cartesians):
            subplot_num = int("{}{}".format(subplot_num1,i+1))
            #print i
            ax = fig.add_subplot(subplot_num, projection='3d')       
            ax.plot(value[:,0], value[:,1], value[:,2])
            ax.set_xlabel('X', fontsize = 20)
            ax.set_ylabel('Y', fontsize = 20)
            ax.set_zlabel('Z', fontsize = 20)
            ax.set_title(scores[i])
        fig.suptitle("Plot for {}".format(gesture), fontsize = 20)
        plt.savefig("{}/{}3d.pdf".format(segmentPath, expertise), dpi = 100)


    def plotGraphAll(self,constraints = None, novices=None, intermediary=None, experts= None, segment = None, datatype = None, subplots = None):
        fig = plt.figure()
        subplotnum1 =int("{}1".format(subplots))
        print "plotting for {} with span {}".format(datatype, subplots)
        fig.set_size_inches(30,20)
        imagePath = self.dataPath + "/figures/" + datatype
        segmentPath = imagePath + "/%s/"%segment
        labelKeys = self.getLabels(datatype)
        print (len(experts))        
        if not os.path.exists(segmentPath):
            os.makedirs(segmentPath)
    
        for i in range(int(subplots)):
            plotNum = int("{}{}".format(subplotnum1, i+1))
            ax = fig.add_subplot(plotNum)
      
            ax.set_xlabel('Time', fontsize = 20)
            ax.set_ylabel(labelKeys[i], fontsize = 20)
            if not novices == []:
                for novice in novices:
                    ax.plot(novice[:,i], color = "red") 
                    pass
            if not intermediary == []:
                for interm in intermediary:
                    ax.plot(interm[:,i], color = "blue")        
                    pass
            if len(experts)> 0:
                x_max = []     
                x_min = []       
                for expert in experts:
                    ax.plot(expert[:,i], color = "green")        
                    x_max.append(np.amax(expert[:,i]))
                    x_min.append(np.amin(expert[:,i]))                
                xmax0 = ax.get_xlim()
                x0 = max(x_max)*np.ones(int(xmax0[1]))
                x1 = min(x_min)*np.ones(int(xmax0[1]))
                constraints.append(max(x_max))
                constraints.append(max(x_min))
                ax.plot(x0, color = "black", linestyle = "--", linewidth =2)
                ax.plot(x1, color = "black", linestyle = "--", linewidth =2)

        red_patch = mpatches.Patch(color = 'red', label = 'novices')
        blue_patch = mpatches.Patch(color = 'blue', label = 'intermediary')
        green_patch = mpatches.Patch(color = 'green', label = 'expert')
        black_patch = mpatches.Patch(color = 'black', label = 'constraints')
        plt.legend(handles= [red_patch, blue_patch, green_patch, black_patch], fontsize=20)
        plt.savefig("{}/expertAllPlots.pdf".format(segmentPath), dpi = 100)
        plt.close()

    def getConstraints(self, cartesians):
        for i in range (cartesians.shape[0]):
           pass 
    def makeSegments(self, cartesians, transcript, segmentPath):
        """
        This function segments the trajectory based on the boundaries obtained from the transcript file
        """
        uniqueGestures =  set(transcript[:,2])
        tree = list(transcript[:,2])
        segmentDict = {}
        countSegment = {}
        for gesture in uniqueGestures:
            segmentDict[gesture] = tree.count(gesture)
            countSegment[gesture] = 0
        prev = "start"
        for i in range(transcript.shape[0]):
            segmentDict[transcript[i][2]] = segmentDict[transcript[i][2]] + 1
            segment = cartesians[transcript[i][0]:transcript[i][1]]
            pickleFile = "{}/{}.p".format(segmentPath,transcript[i][2])            
            externals.joblib.dump(segment, "{}/{}->{}.p".format(segmentPath,transcript[i][2], countSegment[transcript[i][2]]))
            prev = transcript[i][2]
            countSegment[transcript[i][2]] +=1
            #print (segmentDict[transcript[i][2]])

    def gatherSegments(self, key, key1):
        demonstrationPath = self.dataPath + key
        globPath =  demonstrationPath + "/**/"     
        allSegments = []
        segmentPath = []
        for name in glob.glob(globPath):
            _filename = "{}{}/*".format(name, key1)
            for pickle in glob.glob(_filename):
                segmentPath.append(pickle)
                allSegments.append(pickle.split("/")[len(pickle.split("/"))-1])
        uniqueSegments = set(allSegments)
        segmentDict = {}
        for seg in uniqueSegments:   
            segmentDict[seg] = [] #allSegments.count(seg)
        for i in range(len(segmentPath)):
            segmentDict[allSegments[i]].append(segmentPath[i])
        return uniqueSegments, segmentDict

    def normalizeSegments(self, segments):
        normalizedSegments = segments.copy()
        normalizedSegments = normalizedSegments - segments[0]
        return normalizedSegments

    def mapSegmentstoScores(self, uniqueSegments, segmentPath, scores, key, span):
        """
        Maps segments to scores and sends them for plots
        """
        kinDict = {}        
        constraintDict = {}
        for seg in uniqueSegments:
            kinDict[seg] = []
            novice_segments = []
            intermediary_segments = []
            expert_segments = []
            novice_scores = []
            intermediary_scores = []
            expert_scores = []
            constraintDict[seg] = []
            for minseg in segmentPath[seg]:
                kin = externals.joblib.load(minseg)
                kinDict[seg].append(kin.shape)
                name = minseg.split("/")[len(minseg.split("/"))-3]
                if scores[name][0][2] >20:
                    expert_segments.append(kin)
                    expert_scores.append(scores[name][0][2])
                elif scores[name][0][2]>10:
                    intermediary_segments.append(kin)
                    intermediary_scores.append(scores[name][0][2])
                else:
                    novice_segments.append(kin)
                    novice_scores.append(scores[name][0][2])
#            print("checking Score {}".format(novice_scores))
            self.plotGraphAll(constraints = constraintDict[seg], novices = np.array(novice_segments), intermediary = np.array(intermediary_segments), experts = np.array(expert_segments), segment=minseg.split("/")[len(minseg.split("/"))-1],datatype = key, subplots = span)            
            """
            self.compareTrajectories(expert_segments, novice_segments, expert_scores, novice_scores)
            self.plotGraph(novice_segments, novice_scores, minseg.split("/")[len(minseg.split("/"))-1], "novice")                
            self.plotGraph(expert_segments, expert_scores, minseg.split("/")[len(minseg.split("/"))-1], "expert")               
            """     
        self.saveConstraints(constraintDict, key)

    def getLabels(self, key):
        labels= {}
        labels['cartesian'] = ['x', 'y', 'z']
        labels['rotation'] = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']
        labels['linearVelocity'] = ["x'", "y'", "z'"]
        labels['angularVelocity'] = ["alpha","beta'","gamma'"]
        labels['grasperAngle'] = ["theta"]
        return labels[key]

    def saveConstraints(self, constraintsDict, key):
        constraintsPath = self.dataPath + "/constraints/"         
        if not os.path.exists(constraintsPath):
            os.makedirs(constraintsPath)
        externals.joblib.dump(constraintsDict, "{}{}.p".format(constraintsPath, key))

path = os.path.abspath(os.path.dirname(sys.argv[0]))
npds = needlePassing(path)
