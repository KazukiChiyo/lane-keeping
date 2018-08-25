from moviepy.editor import VideoFileClip
import sys

t = sys.argv[1]
clip = VideoFileClip("../project_video.mp4")
clip.save_frame("extracted-"+t+".jpg", t=float(t))
