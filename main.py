import logging
import os
import time
import cv2
from deepHair.Chair import Chair
from deepface.detectors import FaceDetector
from deepface import DeepFace
import yaml
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#TODO 
# - Multithread for multiple chairs and constant video flow
# - improve sampling with subclasses Sample() etc
# - Make yolov4-tiny work

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

print(config["model_name"])

start_model_build = time.perf_counter()
model = DeepFace.build_model(config['model_name'])
face_detector = FaceDetector.build_model(config['detector_backend'])

logger.info(f'Built model in {time.perf_counter()-start_model_build} sec.')
    


def runDeepHair_FaceMatcher(source: str) -> tuple[int, float]:
    chair = Chair([0,2400,800,2400],1, config=config)

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    t1 = 0

    while True:
        t0 = time.time()
        videoTime = int(cap.get(cv2.CAP_PROP_POS_FRAMES))/fps
        ret, frame = cap.read()
        
        #print(frame.shape)
        if ret : 
            cv2.imshow('frame', frame[0:2400,800:2400,:])
        else : break
        if t0-t1>10: 
            t1 = time.time()
            #print(f' affichage {t1-t0}')
            
            chair.update(frame, videoTime, model, face_detector)

            #print('updated')

            #print(time.time()-t1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()



    cv2.destroyAllWindows()
    return chair._Chair__customerID, videoTime

if __name__ =='__main__':
    source = 'true_test.mp4'
    start  = time.time()
    numberOfCustomer, videoTime = runDeepHair_FaceMatcher(source = source )
    print(f'There was {numberOfCustomer} customers in the {source}. It took {time.time()-start} seconds and the video was {videoTime} seconds long ')
