import cv2

img = cv2.imread('github2.png')
ret,thresh2=cv2.threshold(img,10,255,cv2.THRESH_BINARY_INV)  
print(img)
cv2.imshow('s', thresh2)
cv2.imwrite("github.png", thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()
