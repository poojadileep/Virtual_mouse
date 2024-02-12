import cv2
import  numpy as np
import  HandTrackingModule as htm
import  time
import  autopy

###############
wcam, hcam = 640, 480
###############
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    sucess, img = cap.read()


    # 1. Find hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # print(lmList)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) !=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print( x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)
        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (0, wcam), (0, wScr))
            y3 = np.interp(y1, (0, hcam), (0, hScr))
            # 5. Convert Coordinates
            # 6. Smoothen Values
            # 7. Move Mouse
            autopy.mouse.move(x3, y3)
            # 8. Both Index and middle fingers are up : Clicking Mode
            # 9. Find distance between fingers
            # 10. Click mouse if distance short
    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("webcam", img)
    cv2.waitKey(1)




