import os
import  numpy as np
import time, thread, threading

im_numbers = np.arange(1,90,3)

original_path = '/home/noam/analytics/data/split_dir/'
new_path = '/home/noam/analytics/data/'

video_original_paths = [original_path + str(p) + '.mp4' for p in im_numbers]
video_new_paths = [new_path + str(p) + '.mp4' for p in im_numbers]

class MyThread(threading.Thread):
    def run(self):
        os.system('docker-compose up')
        pass


for org, new in zip(video_original_paths, video_new_paths):
    os.system('mv '+str(org)+ ' '+ new_path)
    os.system('docker-compose down')
    # os.system('docker-compose up')

    thread = MyThread()
    thread.daemon = True
    thread.start()

    time.sleep(35)
    os.system('mv '+str(new)+ ' '+ original_path)
    os.system('docker-compose down')