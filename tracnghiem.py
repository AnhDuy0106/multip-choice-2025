import cv2
import imutils
from imutils import contours
import numpy as np
from bientc import *

def get_x(s):
    return s[1][0]
def get_y(s):
    return s[1][1]
def get_h(s):
    return s[1][3]
def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]

def crop_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    img_canny = cv2.Canny(blurred, 100, 200)

    # find contours
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in descending order
        cnts = sorted(cnts, key=get_x_ver1)

        # loop over the sorted contours
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

            if w_curr * h_curr > 100000:
                # check overlap contours
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)

                # if list answer box is empty
                if len(ans_blocks) == 0:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr

        # sort ans_blocks according to x coordinate
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        return sorted_ans_blocks


def find_area(input,image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Làm mờ ảnh bằng hàm GaussianBlur giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    #Threshol lấy ngưỡng thích ứng và độ giãn nở để hiển thị các tính năng chính của hình ảnh
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    #Tim khung ben ngoai de tach van ban khoi nen
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #sắp xếp các counter theo diện tích
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    approx = cv2.approxPolyDP(contours[input], 0.008 * cv2.arcLength(contours[input], True), True)
    rect = cv2.minAreaRect(contours[input])
    box = cv2.boxPoints(rect) 

    #Thuc hien transform de xoay van ban
    corner = find_corner_by_rotated_rect(box,approx)
    image = four_point_transform(image,corner)
    wrap = four_point_transform(thresh,corner)
    #resize vùng báo danh
    width = int(image.shape[1] * 2)
    height = int(image.shape[0] * 2)
    dim = (width, height)
    image2=cv2.resize(image.copy(),dim,interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    blurred_2 = cv2.GaussianBlur(gray_img, (9, 9), 0)
    thresh_2 = cv2.adaptiveThreshold(blurred_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #cv2.imshow("Anh sau buoc 4",thresh_2)
    #cv2.waitKey()
    #trả về ảnh nhị phân vùng số báo danh sau khi phóng to
    return thresh_2

def findcorrect(thresh_2):
    contours = cv2.findContours(thresh_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    tickcontours = []
    for c in contours:
        approx1 = cv2.approxPolyDP(c, 0.001 * cv2.arcLength(c, True), True)
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        area = cv2.contourArea(c)
        if w >= 30 and h >= 29 and 0.75 <= ar <= 1.2  and len(approx1) > 7:
            tickcontours.append(c)
    tickcontours = sort_contours(tickcontours, method="left-to-right")[0]
    correct = '' # bien luu so bao danh
    a=0
    count = 0 # xac dinh vi tri to mau tren moi cot
    for (q, i) in enumerate(np.arange(0, len(tickcontours), 10)):
        color = (100,a,1)
        a+=50
        # Sap xep cac contour theo cot
        cnts = sort_contours(tickcontours[i:i + 10], method="top-to-bottom")[0]
        choice = None
        total = 0
        # Duyet qua cac contour trong cot
        for (j, c) in enumerate(cnts):
            # Tao mask de xem muc do to mau cua contour
            mask = np.zeros(thresh_2.shape, dtype= "uint8")
            cv2.drawContours(mask, [c],-1 , 255, -1)
            mask = cv2.bitwise_and(thresh_2, thresh_2, mask= mask)
            total = cv2.countNonZero(mask)
            # Lap de chon contour to mau dam nhat
            if choice is None or total > choice[0]:
               choice = (total, j)
               # Neu dung Thi to mau xanh
               count = j
               color = (0, 255, 0)
        correct+=str(count)
        
    return correct


def findanswer_1_40(thresh_2, ANSWER_KEY):
    contours = cv2.findContours(thresh_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    tickcontours = []
    
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 30 and h >= 30 and 0.9 <= ar <= 1.3:
            tickcontours.append(c)

   
    tickcontours = sort_contours(tickcontours, method="top-to-bottom")[0]
    correct = 0
    answer_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    for (q, i) in enumerate(np.arange(0, len(tickcontours), 4)):

        cnts = sort_contours(tickcontours[i:i + 4])[0]
        choice = (0, 0)
        total = 0
        
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh_2.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh_2, thresh_2, mask=mask)
            total = cv2.countNonZero(mask)
            if total > choice[0]:
                choice = (total, j)

        current_right = answer_to_index[ANSWER_KEY[q][0]]

        if current_right == choice[1]:
            color = (0, 255, 0)
            correct += 1
        else:
            color = (0, 0, 255)

        # Draw the correct answer
        cv2.drawContours(image, [cnts[current_right]], -1, color, 0)


    return thresh_2, image, correct

def findanswer_1_8(thresh_2, ANSWER_KEY_2):
    contours = cv2.findContours(thresh_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    tickcontours = []
    
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 30 and h >= 30 and 0.8 <= ar <= 1.2:
            tickcontours.append(c)

   
    tickcontours = sort_contours(tickcontours, method="top-to-bottom")[0]
    correct = 0
    answer_to_index = {'D': 0, 'S': 1}

    for (q, i) in enumerate(np.arange(0, len(tickcontours), 2)):

        cnts = sort_contours(tickcontours[i:i + 2])[0]
        choice = (0, 0)
        total = 0
        
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh_2.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh_2, thresh_2, mask=mask)
            total = cv2.countNonZero(mask)
            if total > choice[0]:
                choice = (total, j)

        current_right = answer_to_index[ANSWER_KEY_2[q][0]]

        if current_right == choice[1]:
            color = (0, 255, 0)
            correct += 1
        else:
            color = (0, 0, 255)

        # Draw the correct answer
        cv2.drawContours(image, [cnts[current_right]], -1, color, 1)


    return thresh_2, image, correct

def findanswer_1_6(thresh_2):
    contours = cv2.findContours(thresh_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    tickcontours = []
    for c in contours:
        approx1 = cv2.approxPolyDP(c, 0.001 * cv2.arcLength(c, True), True)
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        area = cv2.contourArea(c)
        if w >= 30 and h >= 29 and 0.75 <= ar <= 1.2  and len(approx1) > 7:
            tickcontours.append(c)
    tickcontours = sort_contours(tickcontours, method="left-to-right")[0]
    correct = '' # bien luu dap an
    a=0
    count = 0 # xac dinh vi tri to mau tren moi cot
    for (q, i) in enumerate(np.arange(0, len(tickcontours), 12)):
        color = (100,a,1)
        a+=50
        # Sap xep cac contour theo cot
        cnts = sort_contours(tickcontours[i:i + 12], method="top-to-bottom")[0]
        choice = None
        total = 0
        # Duyet qua cac contour trong cot
        for (j, c) in enumerate(cnts):
            # Tao mask de xem muc do to mau cua contour
            mask = np.zeros(thresh_2.shape, dtype= "uint8")
            cv2.drawContours(mask, [c],-1 , 255, -1)
            mask = cv2.bitwise_and(thresh_2, thresh_2, mask= mask)
            total = cv2.countNonZero(mask)
            # Lap de chon contour to mau dam nhat
            if choice is None or total > choice[0]:
               choice = (total, j)
               # Neu dung Thi to mau xanh
               count = j
               color = (0, 255, 0)
        correct+=str(count)
        
    return correct

if __name__ == "__main__":
  ANSWER_KEY = { 0: ['A'], 1: ['D'], 2: ['A'], 3: ['A'], 4: ['A'], 5: ['A'], 6: ['A'], 7: ['A'], 8: ['C'], 9: ['B'], 
                   10: ['A'], 11: ['C'], 12: ['A'], 13: ['B'], 14: ['A'], 15: ['C'], 16: ['B'], 17: ['A'], 18: ['C'], 19: ['B'], 
                   20: ['D'], 21: ['D'], 22: ['B'], 23: ['C'], 24: ['C'], 25: ['C'], 26: ['B'], 27: ['A'], 28: ['B'], 29: ['D'], 
                   30: ['C'], 31: ['B'], 32: ['C'], 33: ['B'], 34: ['C'], 35: ['D'], 36: ['C'], 37: ['D'], 38: ['A'], 39: ['C'] }
  
  ANSWER_KEY_2 = {  0: ['D'], 1: ['D'], 2: ['S'], 3: ['S'], 4: ['D'], 5: ['S'], 6: ['D'], 7: ['D'], 8: ['S'], 9: ['S'], 
                   10: ['D'], 11: ['S'], 12: ['S'], 13: ['D'], 14: ['D'], 15: ['S'], 16: ['D'], 17: ['D'], 18: ['D'], 19: ['S'], 
                   20: ['D'], 21: ['D'], 22: ['S'], 23: ['D'], 24: ['S'], 25: ['D'], 26: ['D'], 27: ['S'], 28: ['D'], 29: ['S'], 
                   30: ['S'], 31: ['D'] }

  total_correct = 0
  img = cv2.imread("p2.png")
  image = img.copy()
  thresh_SBD = find_area(18,image)
  thresh_MDT = find_area(25,image)
  madethi    = findcorrect(thresh_MDT)
  sobaodanh  = findcorrect(thresh_SBD)
  #xét từng contour cho phần 1
  thresh_1_10 = find_area(5, image)
  answers_1_10 = [ANSWER_KEY[k] for k in range(0, 10)]
  _, _, ans_1_10 = findanswer_1_40(thresh_1_10, answers_1_10)   
  
  thresh_11_20 = find_area(6, image)
  answers_11_20 = [ANSWER_KEY[k] for k in range(10, 20)]
  _, _, ans_11_20 = findanswer_1_40(thresh_11_20, answers_11_20) 

  thresh_21_30 = find_area(15, image)
  answers_21_30 = [ANSWER_KEY[k] for k in range(20, 30)]
  _, _, ans_21_30 = findanswer_1_40(thresh_21_30, answers_21_30) 

  thresh_31_40 = find_area(8, image)
  answers_31_40 = [ANSWER_KEY[k] for k in range(30, 40)]
  _, _, ans_31_40 = findanswer_1_40(thresh_31_40, answers_31_40) 

  thresh_1_1 = find_area(25,image)
  answers_1_1 = [ANSWER_KEY_2[k] for k in range(0, 4)]
  _, _, ans_1_1 = findanswer_1_8(thresh_1_1, answers_1_1)

  thresh_1_2 = find_area(30,image)
  answers_1_2 = [ANSWER_KEY_2[k] for k in range(4, 8)]
  _, _, ans_1_2 = findanswer_1_8(thresh_1_2, answers_1_2)

  thresh_1_3 = find_area(28,image)
  answers_1_3 = [ANSWER_KEY_2[k] for k in range(8, 12)]
  _, _, ans_1_3 = findanswer_1_8(thresh_1_3, answers_1_3)

  thresh_1_4 = find_area(31,image)
  answers_1_4 = [ANSWER_KEY_2[k] for k in range(12, 16)]
  _, _, ans_1_4 = findanswer_1_8(thresh_1_4, answers_1_4)

  thresh_1_5 = find_area(27,image)
  answers_1_5 = [ANSWER_KEY_2[k] for k in range(16, 20)]
  _, _, ans_1_5 = findanswer_1_8(thresh_1_5, answers_1_5)

  thresh_1_6 = find_area(32,image)
  answers_1_6 = [ANSWER_KEY_2[k] for k in range(20, 24)]
  _, _, ans_1_6 = findanswer_1_8(thresh_1_6, answers_1_6)

  thresh_1_7 = find_area(26,image)
  answers_1_7 = [ANSWER_KEY_2[k] for k in range(24, 28)]
  _, _, ans_1_7 = findanswer_1_8(thresh_1_7, answers_1_7)

  thresh_1_8 = find_area(29,image)
  answers_1_8 = [ANSWER_KEY_2[k] for k in range(24, 32)]
  _, _, ans_1_8 = findanswer_1_8(thresh_1_8, answers_1_8)

  thresh_3_1 = find_area(11, image)
  answers_3_1= findcorrect(thresh_3_1)
  string_3_1 = ''.join(map(str, answers_3_1))

  thresh_3_2 = find_area(9, image)
  answers_3_2= findcorrect(thresh_3_2)
  string_3_2 = ''.join(map(str, answers_3_2))
  
  thresh_3_3 = find_area(10, image)
  answers_3_3= findcorrect(thresh_3_3)
  string_3_3 = ''.join(map(str, answers_3_3))

  thresh_3_4 = find_area(13, image)
  answers_3_4= findcorrect(thresh_3_4)
  string_3_4 = ''.join(map(str, answers_3_4))
  
  thresh_3_5 = find_area(12, image)
  answers_3_5= findcorrect(thresh_3_5)
  string_3_5 = ''.join(map(str, answers_3_5))
  
  thresh_3_6 = find_area(7, image)
  answers_3_6= findcorrect(thresh_3_6)
  string_3_6 = ''.join(map(str, answers_3_6))

  total_correct = ans_1_10 + ans_11_20 + ans_21_30 + ans_31_40 + ans_1_1 + ans_1_2 + ans_1_3 + ans_1_4 + ans_1_5 + ans_1_6 + ans_1_7 + ans_1_8 
  

  string_sbd = ''.join(map(str, sobaodanh))
  string_mdt = ''.join(map(str, madethi))
  cv2.putText(image, f" {string_sbd}", (780, 138), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  cv2.putText(image, f" {string_mdt}", (945, 138), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  cv2.putText(image, f"{total_correct}/40", (880, 41), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  
  cv2.putText(image, f" {string_3_1}", (92, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  cv2.putText(image, f" {string_3_2}", (248, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  cv2.putText(image, f" {string_3_3}", (401, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  cv2.putText(image, f" {string_3_4}", (556, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  cv2.putText(image, f" {string_3_5}", (713, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  cv2.putText(image, f" {string_3_6}", (870, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  
  result_image = cv2.resize(image, (800, 800))
  cv2.imshow("Result", result_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()