import os
import numpy as np
import matplotlib.pyplot as plt
from .tools import import_with_caution, create_folder

class VideoWriter(object):
    def __init__(self, 
            dir_name,
            file_name,
            window_size=(128, 128),
            frame_rate=5):
        self.file_name = file_name
        self.cv2 = import_with_caution('cv2')
        fourcc = self.cv2.VideoWriter_fourcc(*'mp4v') 
        create_folder(dir_name)
        self.image_dir = f'{dir_name}/imgs'
        create_folder(self.image_dir)
        self.ind = 0
        self.window_size = window_size
        self.video_writer = self.cv2.VideoWriter(f'{dir_name}/{file_name}.mp4', fourcc, frame_rate, window_size) 

    def add_image(self, image):
        self.ind += 1
        img_out = self.cv2.resize(image, self.window_size)
        img_write = image.clip(0, 255).astype(np.uint8)
        self.cv2.imwrite(f'{self.image_dir}/frame_{self.ind:06d}.jpg', img_write)
        self.video_writer.write(img_write)

    def clear(self):
        plt.close()
        self.video_writer.release()

if __name__=="__main__":
    writer = VideoWriter("./videos", "demo")
    image1 = np.random.rand(128, 128, 3) * 255
    image2 = np.random.rand(128, 128, 3) * 255
    writer.add_image(image1)
    writer.add_image(image2)
    writer.clear()