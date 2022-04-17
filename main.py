import logging
import multiprocessing
import time

import cv2
import pandas as pd
import yaml
from deepface import DeepFace
from deepface.detectors import FaceDetector
from pandas.errors import EmptyDataError

from deepHair.Chair import Chair
from VideoThread import VideoThread

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
    :param config: dict = {
    :type config: dict
    """

    start_model_build = time.perf_counter()
    model = DeepFace.build_model(config["model_name"])
    face_detector = FaceDetector.build_model(config["detector_backend"])

    logger.info(f"Built model in {time.perf_counter()-start_model_build} sec.")

    chairs: list[Chair] = [
        Chair(camera["chairs"][chair], chair, config) for chair in camera["chairs"]
    ]

    videoThread = VideoThread(camera["source"]).start()

    while not (videoThread.stopped):

        frame = videoThread.frame
        videoTime = videoThread.videoTime

        for chair in chairs:
            chair.update(frame, videoTime, model, face_detector)
            frame = cv2.rectangle(
                frame,
                (chair.AREA[2], chair.AREA[0]),
                (chair.AREA[3], chair.AREA[1]),
                (255, 0, 0),
                4,
            )

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
