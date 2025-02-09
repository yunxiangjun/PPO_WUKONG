import cv2
import numpy as np
import dxcam_cpp as dxcam
import atexit
import time
import datetime
from PIL import Image
import os

# 初始化 dxcam
# time.sleep(3)
width = 448
start_width = 420
height = 448
start_height = 100
camera = None
# 1280 / 2 = 640
# 720 / 2 = 360
def init():
    global camera
    camera = dxcam.create(output_idx=0, output_color="BGRA")
    camera.start(target_fps=30, video_mode=True)
    atexit.register(camera.stop)

def test():
    init()
    time.sleep(3)
    resized_image = get_screen()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image = Image.fromarray(resized_image)
    save_path = os.path.join(f"screenshot_{current_time}.png")
    image.save(save_path)
    # 显示截图
    cv2.imshow('Screenshot', resized_image)
    # 等待用户按键
    cv2.waitKey(0)
    # 关闭窗口
    cv2.destroyAllWindows()

def get_screen():
    frame = camera.get_latest_frame()
    frame = cv2.cvtColor(np.array(frame)[start_height:start_height+height,start_width:start_width+width], cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    return resized_image
