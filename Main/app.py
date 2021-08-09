import cv2 as cv
import numpy as np
from Files.inference import Network
from Files.arguments import get_args
#Global variables

car_d_model_xml = "Models\\vehicle-detection-adas-0002.xml"
car_d_model_bin = "Models\\vehicle-detection-adas-0002.bin"

ped_model_xml = "Models\pedestrian-detection-adas-0002.xml"
ped_model_bin = "Models\pedestrian-detection-adas-0002.bin"

FOURCC = cv.VideoWriter_fourcc('m','p','4','v')

def ped_boxes(frame, result, width, height, color):
    counter = 0
    for box in result[0][0]:
        conf = box[2]
        if conf >= .6:
            counter += 1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    cv.putText(frame, "Detected Pedestrians in Frame: " + str(counter), (50, 85), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
    return frame

def car_boxes(frame, result, width, height, color):
    counter = 0
    for box in result[0][0]:
        conf = box[2]
        if conf >= .6:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            counter += 1
            text = "Car #: " + str(counter)
            cv.putText(frame, text, (xmin + 5, ymin + 40), cv.FONT_HERSHEY_SIMPLEX, 1 , (0,255,0), 2, cv.LINE_AA)
    cv.putText(frame, "Detected Cars In Frame: " + str(counter), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
    return frame


def video_interpreter(args, cap, width, height):
    #Camera mode
    if args.v == 0:
        cam = cap
        camera_capture = cv.VideoWriter('Outputs\Videos\camfeed.mp4', FOURCC, 30, (width,height))
        print("\n\n\n\n------------------\nCamera Mode Detected, Opening live feed window now....\n------------------\n\n\n\n")
        while True:
            flag, frame = cam.read()
            width = int(cam.get(3))
            height = int(cam.get(4))
            cv.imshow('Reference Video: Press ''q'' To End Video', frame)
            if cv.waitKey(25) & 0xFF == ord('q') or flag != True:
                break
            camera_capture.write(frame)
        camera_capture.release()
    else:
        print("\n\n\n\n------------------\nVideo Mode Detected, Interpreting video feed now....\n------------------\n\n\n\n")

def infer_on_video(args):
    # Interpreting video/camera feed and getting basic dimensions
    #_______________________________________________________________________________________
    cap = cv.VideoCapture(args.v)
    cap.open(args.v)
    width = int(cap.get(3))
    height = int(cap.get(4))
    video_interpreter(args, cap, width, height)
    if args.v == 0:
        cap = cv.VideoCapture('Outputs\Videos\camfeed.mp4')

    ped_plugin = Network()
    ped_plugin.load_model(ped_model_xml, ped_model_bin, "CPU")
    ped_input_shape = ped_plugin.get_input_shape()

    car_plugin = Network()
    car_plugin.load_model(car_d_model_xml, car_d_model_bin, "CPU")
    car_input_shape = car_plugin.get_input_shape()
    
    print("\n------------------\nLoading models and making inferences...\n------------------\n\n\n\n\n")
    if args.d == "VPU":
        out = cv.VideoWriter('Outputs\output.mp4', 0x00000021, 30, (width,height))
    if args.d == "CPU":
        out = cv.VideoWriter('Outputs\output.mp4', FOURCC, 30, (width,height))
    while cap.isOpened():
        flag,frame = cap.read()
        if not flag:
            break

        ped_frame = cv.resize(frame, (ped_input_shape[3], ped_input_shape[2]))
        ped_frame = ped_frame.transpose((2,0,1))
        ped_frame = ped_frame.reshape(1, *ped_frame.shape)
        ped_plugin.async_inference(ped_frame)

        car_frame = cv.resize(frame, (car_input_shape[3], car_input_shape[2]))
        car_frame = car_frame.transpose((2,0,1))
        car_frame = car_frame.reshape(1, *car_frame.shape)
        car_plugin.async_inference(car_frame)

        if ped_plugin.wait() == 0:
            result = ped_plugin.extract_output()
            frame = ped_boxes(frame, result, width, height, (255,0,0))
        if car_plugin.wait() == 0:
            result = car_plugin.extract_output()
            frame = car_boxes(frame, result, width, height, (255,0,255))
        out.write(frame)


    print("\n------------------\nInterpretation complete!\n------------------\n")
    cap.release()
    out.release()
    cv.destroyAllWindows()

def main():
    args = get_args()
    infer_on_video(args)
    return 0

if __name__ == '__main__':
    main()