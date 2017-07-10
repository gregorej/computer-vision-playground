from click_and_crop import Cropper, start_event_loop, add_event_loop_listener
import argparse
import cv2
import os
import csv


class VideoCropper(object):

    def __init__(self, window_name):
        self.__cropper = Cropper(window_name)
        self.__video = None
        self.__frame_count = 0
        self.__current_frame = None
        self.__on_crop = lambda crop, i, image: crop

    def __on_key(self, key):
        if key == ord("."):
            ret, frame = self.__video.read()
            if ret:
                self.__frame_count += 1
                self.__current_frame = frame
                self.__cropper.change_image(frame)

    def on_crop(self, crop):
        self.__on_crop = crop

    def start(self, capture):
        self.__video = capture
        self.__cropper.on_crop(lambda crop: self.__on_crop(crop, self.__frame_count, self.__current_frame))
        add_event_loop_listener(lambda key: self.__on_key(key))
        ret, frame = capture.read()
        if not ret:
            return
        self.__frame_count = 1
        self.__current_frame = frame
        self.__cropper.start(frame)


def csv_with_headers(file_path, headers):
    if os.path.isfile(file_path):
        output_csv = open(file_path, 'a')
        csv_writer = csv.writer(output_csv, delimiter=' ')
    else:
        output_csv = open(file_path, 'wb')
        csv_writer = csv.writer(output_csv, delimiter=' ')
        csv_writer.writerow(headers)
    return csv_writer

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, help="Path to the image")
    ap.add_argument("-o", "--output", required=False, default="output")
    args = vars(ap.parse_args())

    output_dir = args["output"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    labels_file_path = output_dir + '/labels.csv'
    labels_writer = csv_with_headers(labels_file_path, ["Frame", "xmin", "ymin", "xmax", "ymax", "Label"])

    video_file_name = args["video"]
    video_file_base_name = os.path.splitext(os.path.basename(video_file_name))[0]

    def handle_crop(crop, frame_no, frame):
        frame_file = video_file_base_name + '_' + str(frame_no) + '.jpg'
        #xmin,ymin,xmax,ymax
        labels_writer.writerow([frame_file, crop[0][0], crop[0][1], crop[1][0], crop[1][1], "car"])
        cv2.imwrite(output_dir + '/' + frame_file, frame)

    cap = cv2.VideoCapture(video_file_name)
    cropper = VideoCropper("video")
    cropper.on_crop(handle_crop)
    cropper.start(cap)

    start_event_loop()

    # close all open windows
    cv2.destroyAllWindows()
