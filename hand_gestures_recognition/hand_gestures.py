from json.tool import main
import cv2
from cvzone.HandTrackingModule import HandDetector

def count_fingers():

    cap = cv2.VideoCapture(0)

    #detect hands
    detector = HandDetector(maxHands=2,detectionCon=0.8)

    while True:
        success, frame = cap.read()

        hands, frame = detector.findHands(frame)
        
        if hands:
            #get landmark list
            lmList = hands[0]['lmList']

            #first finger up
            if lmList[8][1]<lmList[7][1] and lmList[12][1]>lmList[10][1]:
                yield 1
            
            #two fingers up
            elif lmList[12][1]<lmList[10][1] and lmList[8][1]<lmList[7][1]:
                yield 2 

        # cv2.imshow('Video',frame)
        k = cv2.waitKey(1)

        if k==27:
            cv2.destroyAllWindows()
            break

if __name__ == 'main':
    count_fingers()