import datetime
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

global confThreshold

confThreshold = 0.4  # Confidence threshold


def load_model():
    modelConfiguration = "models/yolov4.cfg"
    modelWeights = "models/yolov4.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    return net


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(frame, outs):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classId == 0:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    return boxes, confidences


def process_frame(net, frame) -> tuple[list, list, list, list]:

    nmsThreshold = 0.4  # Non-maximum suppression threshold
    inpWidth = 416  # Width of network's input image
    inpHeight = 416  # Height of network's input image

    x_t = []
    y_t = []
    lengths_t = []
    widths_t = []

    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False
    )

    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers

    outs = net.forward(getOutputsNames(net))

    boxes, confidences = postprocess(frame=frame, outs=outs)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    boxes = [boxes[j] for j in indices]
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        x_t.append((xA + xB) / 2.0)
        y_t.append((yA + yB) / 2.0)
        lengths_t.append(np.abs(xB - xA))
        widths_t.append(np.abs(yB - yA))

    return x_t, y_t, widths_t, lengths_t


def video2data(folder_path: str, output_path: str):

    net = load_model()

    video_files = [
        (f"{folder_path}/{filename}", int(filename.split("_")[1].split(".")[0]))
        for filename in os.listdir(folder_path)
        if (filename.endswith("mp4") and not filename.startswith('.'))
    ]
    video_files.sort(key=lambda x: x[1])

    timestamps = []
    xs = []
    ys = []
    lengths = []
    widths = []

    for video_file_path, _ in video_files:
        i = 0
        time_info = video_file_path.split("/")[-1]
        year = int(time_info[:4])
        month = int(time_info[4:6])
        day = int(time_info[6:8])
        hour = int(time_info[9:11])
        minute = int(time_info[11:13])
        second = int(time_info[13:15])

        start = datetime.datetime(
            year=year, month=month, day=day, hour=hour, minute=minute, second=second
        )

        cap = cv2.VideoCapture(video_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = int(20 * fps)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(
            total=video_length // step,
            desc=f'Video2data | {video_file_path.split("/")[-1]}',
        )
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            
            if i % step == 0:
                for x, y, width, length in zip(*process_frame(net, frame)):
                    xs.append(x)
                    ys.append(y)
                    widths.append(width)
                    lengths.append(length)
                    timestamps.append(start + datetime.timedelta(seconds=20 * (i // step)))
                pbar.update(1)
                
            i += 1
            
        cap.release()

    pd.DataFrame(
        {
            "timestamps": timestamps,
            "x": xs,
            'y' : ys,
            "lengths": lengths,
            "widths": widths,
        }
    ).to_csv(output_path)


if __name__ == "__main__":

    folder_path = "/Volumes/WD_BLACK/tapo_5_06/constantin/20220628"
    output_path = "20220628.csv"
    video2data(folder_path=folder_path, output_path=output_path)
