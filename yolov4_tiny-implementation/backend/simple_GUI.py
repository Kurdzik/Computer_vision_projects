import cv2
import numpy as np


class Option:
    def __init__(self, frame, coordinates, title, title_cords, color=(0, 255, 0)):
        self.coordinates = coordinates
        cv2.fillPoly(frame, coordinates, color)
        cv2.putText(frame, title, title_cords, fontFace=cv2.QT_FONT_NORMAL, fontScale=0.5, color=(0, 0, 0),
                    thickness=1)


class Mark:
    def __init__(self, frame, predicted_class, class_to_find, flag, x, y, w, h, score, show_prob=False,
                 color=(0, 255, 0)):
        if predicted_class == class_to_find and flag:
            # Draw class description
            cv2.putText(frame, predicted_class.capitalize(), org=(x, y - 10), fontFace=cv2.QT_FONT_NORMAL, fontScale=1,
                        color=color, thickness=1)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            

            if show_prob:
                cv2.putText(frame, (str(round(score * 100, 1)) + '%'), org=(x + w - 80, y), fontFace=cv2.QT_FONT_NORMAL,
                            fontScale=1, color=(0, 255, 0), thickness=1)


class Positions:
    def prob(self):
        return np.array([[(110, 10), (250, 10), (250, 30), (110, 30)]])

    def close(self):
        return np.array([[(10, 10), (100, 10), (100, 30), (10, 30)]])

    def all(self):
        return np.array([[(10, 40), (100, 40), (100, 60), (10, 60)]])

    def obj1(self):
        return np.array([[(10, 70), (100, 70), (100, 90), (10, 90)]])

    def obj2(self):
        return np.array([[(10, 100), (100, 100), (100, 120), (10, 120)]])

    def obj3(self):
        return np.array([[(10, 130), (100, 130), (100, 150), (10, 150)]])

    def obj4(self):
        return np.array([[(10, 160), (100, 160), (100, 180), (10, 180)]])

    def obj5(self):
        return np.array([[(10, 190), (100, 190), (100, 210), (10, 210)]])


def mark_all(frame, predicted_class, x, y, w, h, score, show_prob=False):
    cv2.putText(frame, predicted_class.capitalize(), org=(x, y - 10), fontFace=cv2.QT_FONT_NORMAL, fontScale=1,
                color=(0, 255, 0), thickness=1)

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    if show_prob:
        cv2.putText(frame, (str(round(score * 100, 1)) + '%'), org=(x + w - 90, y), fontFace=cv2.QT_FONT_NORMAL,
                    fontScale=1, color=(0, 255, 0), thickness=1)
