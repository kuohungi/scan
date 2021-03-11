import cv2
import numpy as np

cap = cv2.VideoCapture("img/book.mp4")

widthImg = 405
heightImg = 480

def preProcessing(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #模糊化
    blur_img = cv2.GaussianBlur(gray_img,(5,5),1)
    #輪廓線
    canny_img = cv2.Canny(blur_img,200,200)
    kernel = np.ones((5,5))
    #擴張
    dial_img = cv2.dilate(canny_img,kernel,iterations=2)
    #侵蝕
    thres_img = cv2.erode(dial_img,kernel,iterations=1)
    return thres_img

#擷取邊線並繪製
def getContours(img):
    biggest = np.array([])
    maxArea = 0
    #僅檢測外輪廓 方法為儲存所有的輪廓點
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #面積超過多少才畫框線      
        if area>5000:
            #在contour_img上畫線
            
            #找輪廓弧長 封閉為True
            peri = cv2.arcLength(cnt,True)
            #算有多少拐彎點 乘上分辨率
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            #找最大面積四邊形
            if area > maxArea and len(approx) == 4:
                #cv2.drawContours(contour_img, cnt, -1, (255, 255, 0), 2)
                biggest = approx
                maxArea = area
    #找四個角的點點
    cv2.drawContours(contour_img, biggest, -1, (255, 255, 0), 20)
    return biggest

#處理不同角度四個點轉換問題
def reorder (myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    # print("add", add)
    # 重新排序 最小的放最前面 最大的放最後面
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print("NewPoints",myPointsNew)
    return myPointsNew

#輪廓處理
def getWarp(img,biggest):
    biggest = reorder(biggest)
    #擷取的四個點
    pts1 = np.float32(biggest)
    #想將四個點呈現的位置
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    # 圖片扭曲轉換
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    
    #裁減並修正邊邊
    cropped_img = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    cropped_img = cv2.resize(cropped_img,(widthImg,heightImg))
    return cropped_img

#堆疊
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

while True:
    success, img = cap.read()
    img = cv2.resize(img,(widthImg,heightImg))
    contour_img = img.copy()
    #侵蝕
    thres_img = preProcessing(img)
    #找最大四邊形
    biggest = getContours(thres_img)
    #print(biggest)
    warped_img = getWarp(img,biggest)
    
    # cv2.imshow("Result", warped_img)
    # cv2.imshow("img",contour_img)
    stackedImages = stackImages(0.6,[contour_img, warped_img])
    cv2.imshow("scan", stackedImages)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()