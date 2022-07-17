import logging
import multiprocessing
import time

import cv2
import pandas as pd
import yaml
import os
from deepface import DeepFace
from deepface.detectors import FaceDetector
from pandas.errors import EmptyDataError
import numpy as np




from deepHair.Chair import Chair

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO
# - Multithread for multiple chairs and constant video flow
# - improve sampling with subclasses Sample() etc
# - Make yolov4-tiny work

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

cameras = config["cameras"]


def run():
    """
    > For each camera in the config file, start a process that will run the `process_task` function with
    the camera's config and the main config as arguments
    """
    Processes: list[multiprocessing.Process] = []

    for camera in cameras:
        Processes.append(
            multiprocessing.Process(
                target=process_task, args=(config["cameras"][camera], config)
            )
        )
        Processes[-1].start()

    for process in Processes:
        process.join()


def process_task(camera: dict, config: dict):
    """
    It takes a camera and a config, builds a model and a face detector, then starts a video thread and a
    video shower, and while the video thread is running, it updates the chairs with the frame, the video
    time, the model, and the face detector, and then it draws a rectangle around the chair area and
    shows the frame

    :param camera: dict
    :type camera: dict
    :param config: dict 
    :type config: dict
    """

    start_model_build = time.perf_counter()
    model = DeepFace.build_model(config["model_name"])
    face_detector = FaceDetector.build_model(config["detector_backend"])

    logger.info(f"Built model in {time.perf_counter()-start_model_build} sec.")

    chairs: list[Chair] = [
        Chair(camera["chairs"][chair], chair, config) for chair in camera["chairs"]
    ]

    COLORS = [(255, 0, 0),(0, 0, 255)]
    #videoThread = VideoThread(camera["source"]).start()
    videoTime = 0
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    files = [[cv2.imread(f'samples/{filename}'), int(filename.split('_')[1].split('.')[0])]  for filename in os.listdir('samples') if filename.endswith('jpg')]
    files.sort(key= lambda x : x[1])

    for img,_ in files:

        frame = img
        videoTime +=30

# Detecting people in the frame.
        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # boxes, weights = hog.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.0 )

        # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        # for (xA, yA, xB, yB) in boxes:
        #     # display the detected boxes in the colour picture
        #     cv2.rectangle(frame, (xA, yA), (xB, yB),
        #                     (0, 255, 0), 2)

        for chair in chairs:
            chair.update(frame, videoTime, model, face_detector)
            frame = cv2.rectangle(
                frame,
                (chair.AREA[2], chair.AREA[0]),
                (chair.AREA[3], chair.AREA[1]),
                COLORS[int(chair._Chair__isOccupied)],
                4,
            )
            cv2.putText(frame, str(chair.id),(chair.AREA[3], chair.AREA[1]), fontFace=cv2.FONT_ITALIC, fontScale=1, color=(0, 255, 0),thickness=3 )

        cv2.imshow("frame", frame)
        cv2.waitKey(1)


def main():
    """
    It reads the number of customers in the database, runs the main function, and then reads the number
    of customers in the database again. It then prints the difference between the two numbers, and the
    time it took to run the main function
    """
    start = time.perf_counter()
    try:
        df = pd.read_csv(config["customers file"])
        initial_nb_customer = df.shape[0]
    except EmptyDataError:
        initial_nb_customer = 0
    run()
    try:
        df = pd.read_csv(config["customers file"])
        nb_customer = df.shape[0]
    except EmptyDataError:
        nb_customer = 0
    print(
        f"There was {nb_customer-initial_nb_customer} customers during execution. It took {time.perf_counter()-start} "
    )


if __name__ == "__main__":
    main()
