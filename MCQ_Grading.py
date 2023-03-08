import cv2 # openCV 
import numpy as np
import utils
import os 

path = utils.getPath() + 'Answer Sheet Images' # getting the path for answersheet image folder
sourceImages = [] # list for collecting answersheet images
imageNames = [] # list for collecting answersheet image names
dirContents = os.listdir(path) # listing directory items of 'Answer Sheet Images'
for dc in dirContents: # loop through directory items and append them to the list
    currentImage = cv2.imread(f'{path}/{dc}')
    sourceImages.append(currentImage)
    imageNames.append(os.path.splitext(dc)[0]) # remove file extention of the file
# print(imageNames)

questions = int(utils.getR()) # getting number of questions from the text file
choices = int(utils.getC()) # getting number of choices from the text file
ans = utils.getAnswerList() # getting answer indexes from the text file
w = 50 * choices # fixing warp image width with number of choices
h = 50 * questions # fixing warp image height with number of questions

for pathImg, regNo in zip(dirContents, imageNames): # loop through each image for do the marking process
    pathImg = path + '\\' + pathImg  # creating file path for image
    print(regNo)

    # reading and pre-processing image
    # reading image
    img = cv2.imread(pathImg)
    imgContour = img.copy()
    imgBiggestCont = img.copy()
    # pre processing image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    # find contours and draw them
    # finding contours
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # drawing contours
    cv2.drawContours(imgContour, contours, -1, (207, 79, 0), 2)

    # find rectangles
    rectCon = utils.rectContour(contours)
    # finding corner points (x, y values) of the biggest rectangle
    biggestContour = utils.getCornerPoints(rectCon[0])
    # print(biggestContour)

    if biggestContour.size != 0: # if a rectangle detected
        cv2.drawContours(imgBiggestCont, biggestContour, -1, (0, 0, 255), 10)
        biggestContour = utils.re_order(biggestContour) # re-ordering the points list in understandable way for 4 points

        pt1 = np.float32(biggestContour) # converting to float32
        pt2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        matrix = cv2.getPerspectiveTransform(pt1, pt2) # warping the biggest rectangle to fixed width and height
        imgWarpColored = cv2.warpPerspective(img, matrix, (w, h)) # creating color image with the warp perspective

        # convert to threshold
        # converting image to gray
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

        # converting grayed image to threshold
        imgThreshold = cv2.threshold(imgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]

        boxes = utils.splitBoxes(imgThreshold) # splitting the images into boxes to check whether they are marked or not
        # cv2.imshow("2", boxes[2])
        # cv2.imshow("3", boxes[3])
        # cv2.waitKey(0)
        # print(cv2.countNonZero(boxes[2]), cv2.countNonZero(boxes[3]))

        # getting non-zero pixel values of each boxes (white pixels)
        pixelValue = np.zeros((questions, choices)) # creating empty array for storing pixel count for each answer circle box
        countC = 0
        countR = 0
        # in threshold dark pixel becomes white pixels, and light pixels become black pixels
        # black = zero pixels, white = non-zero pixesls
        for image in boxes: # loop through each box to find number of non-zero pixel values
            totalPixels = cv2.countNonZero(image) # counting non-zero pixels (whites)
            if totalPixels>1000:
                pixelValue[countR][countC] = totalPixels # locating non-zero pixel count in corresponding locations in pixelValue

            countC += 1
            if countC == choices:
                countR += 1
                countC = 0

        print(pixelValue)
        # finding index values of the marked circles
        maxIndexes = []
        for x in range (0, questions):
            arr = pixelValue[x]
            if np.count_nonzero(arr) == 1:
                # print(np.count_nonzero(arr))
                maxIndexVal = np.where(arr == np.amax(arr))
                # print (maxIndexVal[0])
                maxIndexes.append(maxIndexVal[0][0])
            else:
                # print(np.count_nonzero(arr))
                maxIndexes.append(choices+1)

        # Grading
        correctAnsCount = 0
        for x in range (0, questions):
            if ans[x] == maxIndexes[x]: # if correct answer indexes matching with the marked indexes
                correctAnsCount += 1 # counting correct answers

        # print(correctAnsCount)
        utils.write2csv(regNo, correctAnsCount) # writing reg_no and results to .csv file


        # cv2.imshow('Image', imgThreshold)
        # cv2.waitKey(0)
