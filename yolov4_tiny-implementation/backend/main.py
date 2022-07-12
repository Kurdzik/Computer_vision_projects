import cv2
import pandas as pd
from backend.simple_GUI import Option, Mark, Positions, mark_all

p = Positions()

# Load file with classes
x = pd.read_csv('backend/model/dnn_model/classes.txt', header=None).reset_index().rename({0: 'class'}, axis=1)
class_dict = dict(list(zip(x.iloc[:, 0], x.iloc[:, 1])))

# Import and instantiate model

net = cv2.dnn.readNet('backend/model/dnn_model/yolov4-tiny.weights', 'backend/model/dnn_model/yolov4-tiny.cfg')
# net = cv2.dnn.readNet('yolov3_model/yolov3.weights', 'yolov3_model/yolov3.cfg')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(320, 320))

# Enable live feed from camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

# Global vars
current_objs = ('All')
flag_close = False
flag_prob = False
flag_all = True
flag_1 = False
flag_2 = False
flag_3 = False
flag_4 = False
flag_5 = False



def on_left_button_click(e, x, y, arg1, arg2):
    global flag_close, \
        flag_prob,\
        flag_all, \
        flag_1, \
        flag_2, \
        flag_3, \
        flag_4, \
        flag_5
        

    if e == cv2.EVENT_LBUTTONDOWN:

        poly_close = p.close()
        poly_prob = p.prob()
        poly_all = p.all()
        poly_obj1 = p.obj1()
        poly_obj2 = p.obj2()
        poly_obj3 = p.obj3()
        poly_obj4 = p.obj4()
        poly_obj5 = p.obj5()
        

        is_within_close = cv2.pointPolygonTest(poly_close, (x, y), False)
        if is_within_close > 0:
            flag_close = True

        is_within_prob = cv2.pointPolygonTest(poly_prob, (x, y), False)
        if is_within_prob > 0:
            if flag_prob:
                flag_prob = False
            else:
                flag_prob = True

        is_within_obj1 = cv2.pointPolygonTest(poly_obj1, (x, y), False)
        if is_within_obj1 > 0:
            if flag_1:
                flag_1 = False                
            else:
                flag_1 = True
                
        is_within_all = cv2.pointPolygonTest(poly_all, (x, y), False)
        if is_within_all > 0:
            if flag_all:
                flag_all = False
            else:
                flag_all = True

        is_within_obj2 = cv2.pointPolygonTest(poly_obj2, (x, y), False)
        if is_within_obj2 > 0:
            if flag_2:
                flag_2 = False
            else:
                flag_2 = True

        is_within_obj3 = cv2.pointPolygonTest(poly_obj3, (x, y), False)
        if is_within_obj3 > 0:
            if flag_3:
                flag_3 = False
            else:
                flag_3 = True

        is_within_obj4 = cv2.pointPolygonTest(poly_obj4, (x, y), False)
        if is_within_obj4 > 0:
            if flag_4:
                flag_4 = False
            else:
                flag_4 = True

        is_within_obj5 = cv2.pointPolygonTest(poly_obj5, (x, y), False)
        if is_within_obj5 > 0:
            if flag_5:
                flag_5 = False
            else:
                flag_5 = True


cv2.namedWindow('Video')
cv2.setMouseCallback('Video', on_left_button_click)


class Detector:

    def detect(obj1,obj2,obj3,obj4,obj5,detection_treshold=0.2):
        # Read camera feed and run a model on each frame
        while True:
            ret, frame = cap.read()

            # Select classes to detect
            class_ids, score, bounding_box = model.detect(frame, confThreshold=detection_treshold)

            # NMS (non maximum compression) - we need to set this up to avoid duplicated detections
            bounding_box = list(bounding_box)
            score = list(map(float, score.reshape(1, -1)[0]))
            indices = cv2.dnn.NMSBoxes(bboxes=bounding_box, scores=score, score_threshold=detection_treshold, nms_threshold=0.5)

            for i in indices:
                box = bounding_box[i]
                x, y, w, h = box[0], box[1], box[2], box[3]

                Mark(frame, class_dict[class_ids[i]], f'{obj1}', flag_1, x, y, w, h, score[i], show_prob=flag_prob)
                Mark(frame, class_dict[class_ids[i]], f'{obj2}', flag_2, x, y, w, h, score[i], show_prob=flag_prob)
                Mark(frame, class_dict[class_ids[i]], f'{obj3}', flag_3, x, y, w, h, score[i], show_prob=flag_prob)
                Mark(frame, class_dict[class_ids[i]], f'{obj4}', flag_4, x, y, w, h, score[i], show_prob=flag_prob)
                Mark(frame, class_dict[class_ids[i]], f'{obj5}', flag_5, x, y, w, h, score[i], show_prob=flag_prob)

                if flag_all:
                    mark_all(frame=frame, predicted_class=class_dict[class_ids[i]], x=x, y=y, w=w, h=h, score=score[i],
                            show_prob=flag_prob)

            
            if flag_close:
                break

            # Buttons
            Option(frame, p.close(), 'Close', (13, 25), color=(0, 0, 200))
            
            if flag_prob:
                Option(frame, p.prob(), 'Show Probability', (113, 25), color=(0, 0, 150))
            if not flag_prob:
                Option(frame, p.prob(), 'Show Probability', (113, 25), color=(0, 0, 200))

            if flag_all:
                Option(frame, p.all(), 'All', (13, 55),color=(0,150,0))
            if not flag_all:
                Option(frame, p.all(), 'All', (13, 55))
            
            if flag_1:
                Option(frame, p.obj1(), f'{obj1.capitalize()}', (13, 85),color=(0,150,0))
            if not flag_1:
                Option(frame, p.obj1(), f'{obj1.capitalize()}', (13, 85))

            if flag_2:
                Option(frame, p.obj2(), f'{obj2.capitalize()}', (13, 115),color=(0,150,0))
            if not flag_2:
                Option(frame, p.obj2(), f'{obj2.capitalize()}', (13, 115))
            
            if flag_3:
                Option(frame, p.obj3(), f'{obj3.capitalize()}', (13, 145),color=(0,150,0))
            if not flag_3:
                Option(frame, p.obj3(), f'{obj3.capitalize()}', (13, 145))
            
            if flag_4:
                Option(frame, p.obj4(), f'{obj4.capitalize()}', (13, 175),color=(0,150,0))
            if not flag_4:
                Option(frame, p.obj4(), f'{obj4.capitalize()}', (13, 175))
            
            if flag_5:
                Option(frame, p.obj5(), f'{obj5.capitalize()}', (13, 205),color=(0,150,0))
            if not flag_5:
                Option(frame, p.obj5(), f'{obj5.capitalize()}', (13, 205))


            cv2.imshow("Video", frame)
            cv2.waitKey(1)
