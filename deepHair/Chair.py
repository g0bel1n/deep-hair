import itertools
import logging
from datetime import datetime

import numpy as np
import yaml
from deepface import DeepFace
from deepface.commons import distance as dst
from deepface.detectors import FaceDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def noStoredFace(face_storage: list[float]) -> bool:
    """
    > This function returns True if there are no faces stored in the face storage

    :param face_storage: a list of faces that have been detected and stored
    :return: A boolean value.
    """

    return not face_storage


class Chair:
    def __init__(self, AREA: list[int], id: int, config: dict) -> None:
        self.AREA = AREA
        self.__isOccupied = False
        self.image = np.array
        self.id = id

        self.timeLastStore = 0
        self.timeLastSample = 0

        self.SAMPLING_FREQUENCY = config["SAMPLING FREQUENCY"]
        self.STORAGE_FREQUENCY = config["STORAGE FREQUENCY"]
        self.__NecessaryAmountOfStoredFaces = config["NecessaryAmountOfStoredFaces"]
        self.__NecessaryAmountOfSamples = config["NecessaryAmountOfSamples"]

        self.leftCounter: int = 0
        self.nbStoredFacesForCurrentCustomer = 0
        self.idVerified = False
        self.__customerID = 0
        self.samples: list = []
        self.config = config
        self.threshold = dst.findThreshold(
            config["model_name"], config["distance_metric"]
        )

        with open(config["faces_bank_path"], "r") as storage_file:
            self.face_storage = yaml.safe_load(storage_file)
        if self.face_storage is None:
            self.face_storage = {}

    def __newCustomer(self):  # potentially
        """
        The __newCustomer function is a private function that is used to create a new customer
        """
        logger.info("New customer !")
        with open(self.config["customers file"], "a") as file:
            file.write(f"{self.id},{datetime.now()}\n")
        self.__customerID += 1

    def getUpdatedConditions(
        self, newState: bool
    ) -> tuple[bool, bool, bool, bool, bool]:

        stateChanged = self.__isOccupied != newState
        self.__isOccupied = newState
        someoneJustSat = stateChanged and self.__isOccupied and self.leftCounter == 0
        if stateChanged and not self.__isOccupied:
            self.leftCounter = 1
        elif stateChanged:
            self.leftCounter = 0
        elif not self.__isOccupied and self.leftCounter > 0:
            self.leftCounter += 1

        someoneJustLeft = self.leftCounter > self.config["nb_frame_to_consider_left"]

        someoneIsSitting = not stateChanged and self.__isOccupied
        enoughSampleToCheck = len(self.samples) >= self.__NecessaryAmountOfSamples
        mustStoreFace = (
            self.nbStoredFacesForCurrentCustomer < self.__NecessaryAmountOfStoredFaces
        )

        return (
            someoneJustLeft,
            someoneJustSat,
            someoneIsSitting,
            enoughSampleToCheck,
            mustStoreFace,
        )

    def __getAreaFromImg(self, img: np.ndarray) -> np.ndarray:
        """
        Get the area of the image defined by the AREA tuple

        :param img: the image to be cropped
        :type img: np.array
        :return: The image cropped to the area of interest.
        """
        return img[self.AREA[0] : self.AREA[1], self.AREA[2] : self.AREA[3], :]

    def deleteSamples(self):
        """
        It deletes all the files in the sample folder.
        """
        self.samples = []

    def getSample(self, videoTime, model):

        if videoTime - self.timeLastSample > self.SAMPLING_FREQUENCY:
            try:
                face_repr = DeepFace.represent(
                    self.image,
                    model=model,
                    model_name=self.config["model_name"],
                    detector_backend=self.config["detector_backend"],
                )
                self.samples.append(face_repr)
                self.timeLastSample = videoTime
                logger.info(f"{self.id} : Sampled a face")
            except ValueError:
                self.timeLastSample = videoTime
                logger.info(f"{self.id} : Did not detect any face")

    def storeFace(self, videoTime: float, model):
        if videoTime - self.timeLastStore > self.STORAGE_FREQUENCY:
            try:
                face_repr = DeepFace.represent(
                    self.image,
                    model=model,
                    model_name=self.config["model_name"],
                    detector_backend=self.config["detector_backend"],
                )
                self.face_storage[
                    f'stored_{self.id}/at_{datetime.now().strftime("%H_%M_%S")}'
                ] = face_repr
                self.timeLastStore = videoTime
                self.nbStoredFacesForCurrentCustomer += 1
                logger.info(f"{self.id} : Stored a face")
            except ValueError:
                self.timeLastStore = videoTime
                logger.info(f"{self.id} : Did not detect any face")

    def verifyNewCustomer(self) -> bool:

        logger.info("Checking new customer")

        for source, test in itertools.product(self.face_storage.values(), self.samples):
            computed_distance = dst.findCosineDistance(source, test)
            if computed_distance < self.threshold:
                # if there is match
                logger.info(f"{self.id} : The person seated is not a new customer")
                return False

        self.__newCustomer()
        logger.info(f"{self.id} : The person seated is a new customer")
        return True

    def cleanLastPersonVariables(self):
        """
        This function is used to reset the variables used to store the last person's information
        """
        self.idVerified = False

        # Saves new stored faces into the yaml file
        with open(
            self.config["faces_bank_path"], "a", encoding="utf-8"
        ) as storage_file:
            dump = yaml.safe_dump(self.face_storage)
            storage_file.write(dump)

        self.deleteSamples()
        self.nbStoredFacesForCurrentCustomer = 0
        self.leftCounter = 0

    def update(self, img: np.ndarray, videoTime: float, model, face_detector):
        """

        :param img: the image to be processed
        :param videoTime: the current time of the video
        :param model: the model used to detect faces
        :param face_detector: the face detector object
        """

        self.image = self.__getAreaFromImg(img)
        newState = (
            len(
                FaceDetector.detect_faces(
                    face_detector,
                    self.config["detector_backend"],
                    self.image,
                    align=False,
                )
            )
            > 0
        )

        (
            someoneJustLeft,
            someoneJustSat,
            someoneIsSitting,
            enoughSampleToCheck,
            mustStoreFace,
        ) = self.getUpdatedConditions(newState=newState)

        if someoneJustSat:
            logger.info(f"{self.id} : Someone just sat")
            if noStoredFace(self.face_storage):
                logger.info(f"{self.id} : No stored faces")
                # If there is no Stored Face then the customer is obviously a new customer (the first)
                self.__newCustomer()
                self.idVerified = True
            else:
                self.getSample(videoTime, model)

        elif someoneJustLeft:
            logger.info(f"{self.id} : Someone just left")
            if (
                not self.idVerified
                and len(self.samples) >= 2
                and self.verifyNewCustomer()
            ):
                self.storeFace(videoTime, model)
            self.cleanLastPersonVariables()

        elif someoneIsSitting:
            logger.info(
                f"{self.id} : Nb samples : {len(self.samples)}, is verified : {self.idVerified}"
            )
            if not self.idVerified:
                if enoughSampleToCheck:
                    self.verifyNewCustomer()
                    self.idVerified = True
                else:
                    self.getSample(videoTime, model)

            if self.idVerified and mustStoreFace:
                self.storeFace(videoTime, model)
