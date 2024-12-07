import cv2 as cv

filename = "../inputImages/google-chrome-icon.png"
img = cv.imread(filename)
cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window

