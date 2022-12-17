import cv2
import math
import numpy as np

demo = cv2.imread(r'C:\Users\MSI\Desktop\python project\ColorChecker.png',1)
cv2.imshow("demo",demo)

demo2 = cv2.cvtColor(demo,cv2.COLOR_BGR2GRAY)
cv2.imshow("demo2",demo2)

demo3 = cv2.bitwise_not(demo2)
cv2.imshow("demo3",demo3)
