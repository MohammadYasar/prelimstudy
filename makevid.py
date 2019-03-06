import cv2, imageio, os, sys, glob, pandas, math
import moviepy.editor as mpy

class makeVideo:
    def makeVideo(self, _dir):
        globPath = _dir + "*"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        images = []
        images = [img for img in os.listdir(_dir)]
        frame = cv2.imread(os.path.join(_dir, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter("video.avi", fourcc,15, (width, height))
        for name in sorted(glob.glob(globPath)):
            video.write(cv2.imread(name))

        cv2.destroyAllWindows()
        video.release()

    def clipVideo(self, clipName, start, end):
        myclip = mpy.VideoFileClip(clipName)
        print (myclip.fps) # prints for instance '30'
        myclip2 = myclip.subclip(start-10, end+10)
        myclip2.write_videofile(clipName.replace(".avi", "clipped.mp4"), codec='mpeg4')
        #myclip2.write_videofile(clipName.replace(".avi", "clipped.avi"))

    def iterateoverFailures(self, _dir):
        _filename = _dir + "/failure_times.csv"
        _df = pandas.read_csv(_filename)
        for index, row  in _df.iterrows():
            _subject = _dir + "/video/"+ _df.at[index, 'Demonstration'] + "_capture1.avi"
            if not math.isnan(_df.at[index, 'S1']):
                _start = int(_df.at[index, 'S1'])
                _end = int(_df.at[index, 'E1'])

                self.clipVideo(_subject, _start, _end)

_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
mkvd = makeVideo()
#mkvd.makeVideo(_dir)
mkvd.iterateoverFailures(_dir)
