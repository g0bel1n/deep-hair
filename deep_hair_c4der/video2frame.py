import cv2
from tqdm import tqdm
import os

def video2frame():
    i=0
    l = os.listdir('one_day')
    l.sort()
    for file in l:

        cap = cv2.VideoCapture(f'one_day/{file}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = int(20*fps)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(total=length // step)
        while (cap.isOpened()):
                ret, frame = cap.read()
                i+=1
                if ret == False:
                    break
                if i % step == 0:
                    cv2.imwrite(f'one_day_frame/frame_{i}.jpg', frame)
                    pbar.update(1)

        cap.release()

    return True

video2frame()
