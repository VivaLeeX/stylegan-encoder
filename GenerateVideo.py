import numpy as np
import argparse
import os
from moviepy.editor import CompositeAudioClip, VideoFileClip, AudioFileClip
from random import randrange

def render_video(videoName, tmp_dir, num_frames, mode, size, fps, codec, bitrate):

    import PIL.Image
    import moviepy.editor

    def render_frame(t):
        frame = np.clip(np.ceil(t * fps), 1, num_frames)
        image = PIL.Image.open('%s/%08d.png' % (tmp_dir, frame))
        if mode == 1:
            canvas = image
        #else:
        #    canvas = PIL.Image.new('RGB', (2 * src_size, src_size))
        #    canvas.paste(src_image, (0, 0))
        #    canvas.paste(image, (src_size, 0))
        #if size != src_size:
        #    canvas = canvas.resize((mode * size, size), PIL.Image.LANCZOS)
        return np.array(canvas)

    #src_image = PIL.Image.open(src_file)
    #src_size = src_image.size[1]
    duration = num_frames / fps
    #filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.mp4')
    video_clip = moviepy.editor.VideoClip(render_frame, duration=duration)
    video_clip.write_videofile(videoName, fps=fps, codec=codec, bitrate=bitrate)

def AddAudio(lenthOfVideo, inputVideo, outputVideo, audioPath):
    audioArr= ['1.mp3','2.mp3','3.mp3','4.mp3','5.mp3','6.mp3']
    index = randrange(0, len(audioArr))
    if(index > len(audioArr)):
        index = 1

    audioFile = audioPath + '/' + audioArr[index]

    audioStart = randrange(1, 100)

    a = AudioFileClip(audioFile).subclip(audioStart,audioStart + lenthOfVideo)
    video = VideoFileClip(inputVideo)
    newVideo = video.set_audio(a)
    newVideo.write_videofile(outputVideo, audio_codec='aac')

    pass
    
parser = argparse.ArgumentParser(description='Project real-world images into StyleGAN2 latent space')
parser.add_argument('--num_frames', type=int, default=1000, help='Number of optimization steps')
parser.add_argument('--frames_dir', default='.stylegan2-tmp', help='Temporary directory for tfrecords and video frames')
parser.add_argument('--videoName', default=False, help='The out put videoName')
parser.add_argument('--videoNametmp', default=False, help='The out put videoName tmp')
parser.add_argument('--video-mode', type=int, default=1, help='Video mode: 1 for optimization only, 2 for source + optimization')
parser.add_argument('--video-size', type=int, default=1024, help='Video size (height in px)')
parser.add_argument('--video-fps', type=int, default=24, help='Video framerate')
parser.add_argument('--video-codec', default='libx264', help='Video codec')
parser.add_argument('--video-bitrate', default='5M', help='Video bitrate')
parser.add_argument('--audioPath', default='', help='Video bitrate')
args = parser.parse_args()

render_video(args.videoNametmp, args.frames_dir, args.num_frames, args.video_mode,args.video_size, args.video_fps, args.video_codec, args.video_bitrate)
AddAudio(10, args.videoNametmp, args.videoName, args.audioPath)