import argparse
import cv2
import csv
import os
from utils.video import sliding_windows, frames

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", required=True, help="Frame width", type=int)
ap.add_argument("-i", "--height", required=True, help="Frame height", type=int)
ap.add_argument("-v", "--vertical-step", required=False, help="Vertical step", default=10, type=int)
ap.add_argument("-j", "--horizontal-step", required=False, help="Horizontal step", default=10, type=int)
ap.add_argument("-o", "--output-dir", required=False, help="Output directory", default="dataset")
ap.add_argument("video_file")

args = vars(ap.parse_args())

output_dir = args["output_dir"]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

negatives_csv = open(output_dir + '/negatives.csv', 'a')
positives_csv = open(output_dir + '/positives.csv', 'a')
positives_writer = csv.writer(positives_csv, delimiter=' ')
negatives_writer = csv.writer(negatives_csv, delimiter=' ')

video_file = args["video_file"]
video_file_base_name = os.path.splitext(os.path.basename(video_file))[0]

frame_no = 0
for frame in frames(video_file):
    frame_stored = False
    for window in sliding_windows(frame, args["width"], args["height"], args["horizontal_step"], args["vertical_step"]):
        with_sliding_window = frame.copy()
        cv2.rectangle(with_sliding_window, window[0], window[1], (0, 255, 0), 2)
        cv2.imshow("extracting", with_sliding_window)
        key = cv2.waitKey(0) & 0xFF
        frame_file = video_file_base_name + '_' + str(frame_no) + '.jpg'
        if not frame_stored:
            cv2.imwrite(output_dir + '/' + frame_file, frame)
        if key == ord("p"):
            positives_writer.writerow([frame_file, window[0][0], window[0][1], window[1][0], window[1][1]])
            positives_csv.flush()
        elif key == ord("n"):
            negatives_writer.writerow([frame_file, window[0][0], window[0][1], window[1][0], window[1][1]])
            negatives_csv.flush()
        elif key == ord("s"):
            break
    frame_no += 1
