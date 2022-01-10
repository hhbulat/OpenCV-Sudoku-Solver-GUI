import cv2
import numpy as np
import operator
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import time


# CNN deep learning model for digits predictions
classifier = load_model("digit_model.h5")

marge = 4    # margin
case = 28 + 2 * marge  # case size
size_grid = 9 * case # grid size
flag=0
contour_grid = None
maxArea = 0
grid_vector = []
# Sudoku game picture
def import_pics(path):
    im1 = Image.open(path)
    frame = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2BGR)    
    return frame
M = 9 # row/column size 

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,12))
    # cv2.imwrite('solution_pics/thresh.png',cv2.cvtColor(thresh,cv2.COLOR_BGR2RGB))
    # cv2.imwrite('solution_pics/original.png',cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    # ax1.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),ax1.axis('off'),ax1.set_title('Original Frame')
    # ax2.imshow(cv2.cvtColor(thresh,cv2.COLOR_BGR2RGB)),ax2.axis('off'),ax2.set_title('Threshold Image'),plt.show()
    return contours

def contour_customize(contours,maxArea,frame):
     #Frame Contour Tools
    for c in contours:
            area = cv2.contourArea(c)
            if area > 25000:
                peri = cv2.arcLength(c, True)
                polygone = cv2.approxPolyDP(c, 0.01 * peri, True)
                if area > maxArea and len(polygone) == 4:
                    contour_grid = polygone
                    maxArea = area

    cv2.drawContours(frame, [contour_grid], 0, (0, 255, 0), 2)
    points = np.vstack(contour_grid).squeeze()
    points = sorted(points, key=operator.itemgetter(1))

    if points[0][0] < points[1][0]:
        if points[3][0] < points[2][0]:
            pts1 = np.float32([points[0], points[1], points[3], points[2]])
        else:
            pts1 = np.float32([points[0], points[1], points[2], points[3]])

    else:
        if points[3][0] < points[2][0]:
            pts1 = np.float32([points[1], points[0], points[3], points[2]])
        else:
            pts1 = np.float32([points[1], points[0], points[2], points[3]])

    pts2 = np.float32([[0, 0], [size_grid, 0], [0, size_grid], [size_grid, size_grid]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    grid = cv2.warpPerspective(frame, M, (size_grid, size_grid))
    grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
    grid = cv2.adaptiveThreshold(
    grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)
    # cv2.imwrite('solution_pics/grid.png',cv2.cvtColor(grid,cv2.COLOR_BGR2RGB))
    # plt.imshow(cv2.cvtColor(grid,cv2.COLOR_BGR2RGB)),plt.axis('off'),plt.title('Grid'),plt.show()
    return grid

def prediction(grid):
    # Digit Prediction
    for y in range(9):
        line = ""
        for x in range(9):
            # Find Case.
            y2min = y * case + marge
            y2max = (y + 1) * case - marge
            x2min = x * case + marge
            x2max = (x + 1) * case - marge
            img = grid[y2min:y2max, x2min:x2max]
            case_img = img.reshape(1, 28, 28, 1) # for prediction
            # digit prediction
            if case_img.sum() > 10000:
                predict=classifier.predict(case_img) 
                classes=np.argmax(predict)
                line += "{:d}".format(classes)
            else:
                line += "{:d}".format(0)
        grid_vector.append(line)
        
    # print(grid_vector) # 0 == space  
    return grid_vector

def split_chars(grid_vector):
    # Predicted digit vector convert to matrix.
    split_matrix = []
    for row,row_numbers in enumerate(grid_vector):
        try:
            split_rows = list(map(int, row_numbers)) 
        except:
            print(str(row + 1) + "nd line has non-int values.") 

        if len(split_rows) != 9:
            print(str(row + 1) + "nd line has not nine numbers.")

        split_matrix.append(split_rows)
    if row != 8:
        print("matrix contains "+ str(row + 1) +"rows instead of 9") 
    return split_matrix


def show_predict_digits(path,for_empty):
#predict digits writing 9x9 sudoku images.
    empty = cv2.imread(path)

    for y in range(0,9):
        for x in range(0,9):
            if for_empty[y][x] != 0:
                location =(70+(55*x),100+(55*y))
                cv2.putText(empty,str(for_empty[y][x]),location, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # plt.imshow(empty),plt.axis('off'),plt.show()
    # cv2.imwrite('solution_pics/predict_digits.png',cv2.cvtColor(empty,cv2.COLOR_BGR2RGB))
    return empty

# Sudoku Visualizition Func
def puzzle(a):
    for i in range(M):
        for j in range(M):
            print(a[i][j],end = " ")
        print()

# Solve iteration func
def solve(grid, row, col, num):
    
    # query for row
    for x in range(9):
        if grid[row][x] == num:
            return False
    # query for column
    for x in range(9):
        if grid[x][col] == num:
            return False
 
    #query for 3x3 matrix in the game
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True
 
# Main function.
def Sudoku(grid, row, col):
 
    if (row == M - 1 and col == M):
        return True
    if col == M:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return Sudoku(grid, row, col + 1)
    for num in range(1, M + 1, 1): 
        
        if solve(grid, row, col, num):
         
            grid[row][col] = num
            if Sudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False


def gui_predict(path,maxArea=0):
    frame = import_pics(path)
    cnt = preprocess(frame)
    grid = contour_customize(cnt,maxArea,frame)
    grid_vector = prediction(grid)
    split_matrix = split_chars(grid_vector)
    for_empty =  split_matrix

    empty = show_predict_digits('ui/predicted.jpg',for_empty)
    solution = show_predict_digits('ui/solution.jpg',for_empty)

    cv2.imwrite("data/predict.png",empty)
    cv2.imwrite("data/solution.png",cv2.cvtColor(solution,cv2.COLOR_BGR2RGB))
    path = "data/predict.png"
    return path,split_matrix

def gui_solution():
    start_time = time.time()
    _,split_matrix = gui_predict('data/solution.png')
    if (Sudoku(split_matrix, 0, 0)):
        # puzzle(split_matrix)
        success = 'Solution Succesfully!'
    else:
        success = "Solution does not exist:("

    result=split_matrix
    empty = cv2.imread('data/solution.png')

    for y in range(0,9):
        for x in range(0,9):
            if int(grid_vector[y][x]) == 0:
                location =(70+(55*x),100+(55*y))
                cv2.putText(empty,str(result[y][x]),location, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imwrite('data/solution.png',cv2.cvtColor(empty,cv2.COLOR_BGR2RGB))
    path = 'data/solution.png'
    end_time = time.time()
    diff = end_time - start_time
    print(diff)
    return path,diff,success

