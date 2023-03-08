import cv2 # openCV = for computer vison
import numpy as np # numpy = for calculations
import os # interacting with files

# for finding contours if rectangles
def rectContour(contours):
    rectangleContours = []
    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if area > 50:
            perimeter = cv2.arcLength(i, True) # finding perimeter
            approx = cv2.approxPolyDP(i, 0.02*perimeter, True) # finding number of corner points
            # print("Corner points : ", len(approx))
            if len(approx) == 4: # if number of corner points equals to 4 then it's a rectangle
                rectangleContours.append(i) # appending rectangle points to rectangleContours[]

    # print("Rectangle", rectangleContours)
    rectangleContours = sorted(rectangleContours, key=cv2.contourArea, reverse=True)  # sorting rectangles on Area value = Descending order
    return rectangleContours

# getting x, y values of corner points for biggest rectangle
def getCornerPoints(cont):
    perimeter = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * perimeter, True)
    return approx


def re_order(myPoints): # re-ordering the points list in understandable way for 4 points
    myPoints = myPoints.reshape((4, 2))
    mypointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print(myPoints)
    # print(add)
    mypointsNew[0] = myPoints[np.argmin(add)]  # [0, 0] top left
    mypointsNew[3] = myPoints[np.argmax(add)]  # [w, h] bottom right
    diff = np.diff(myPoints, axis=1)
    mypointsNew[1] = myPoints[np.argmin(diff)]  # [w, 0] top right
    mypointsNew[2] = myPoints[np.argmax(diff)]  # [0, h] bottom left
    # print(diff)
    return mypointsNew

def splitBoxes(img): # splitting answer circles to each box
    rows = np.vsplit(img, int(getR()))
    boxes = []
    for r in rows:
        cols = np.hsplit(r, int(getC()))
        for box in cols:
            boxes.append(box)
            # cv2.imshow("Split", box)
    return boxes

# getting current working directory
def getPath():
    workingDir = os.getcwd()
    workingDir = format(workingDir)
    mainPath = workingDir.split('\\')
    workingDir = ''

    for x in range(len(mainPath)-4):
        workingDir = workingDir + mainPath[x]+'\\'

    return workingDir

# getting number of rows (questions)
def getR():
    rcCountPath = getPath() + "rcCount.txt"
    with open(rcCountPath, 'r+') as f:
        rc = f.readlines()[0]
        return rc.split(',')[0]

# getting number of columns (choices)
def getC():
    rcCountPath = getPath() + "rcCount.txt"
    with open(rcCountPath, 'r+') as f:
        rc = f.readlines()[0]
        return rc.split(',')[1]

# reading correct answer indexes from text file
def getAnswerList():
    ansIndexPath = getPath() + "ansIndex.txt"
    ans = []
    with open(ansIndexPath, 'r+') as f:
        ansTempList = f.readlines()
        for x in ansTempList:
            ans.append(int(x))
        return ans

# writing registration number and the results to the .csv file
def write2csv(regNo, mark):
    gradesCSVpath = getPath() + 'Results Sheet\\Grades.csv'
    with open(gradesCSVpath, 'r+') as f:
        data_list = f.readlines()
        name_list = []
        for line in data_list:
            record = line.split(',')
            name_list.append(record[0])
        if regNo not in name_list:
            f.writelines(f'\n{regNo}, {mark}')

