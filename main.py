#采样色块面积
from threading import Thread
import threading
import math, time, queue
import datetime

from networkx.drawing.tests.test_pylab import plt
from tqdm import tqdm
from termcolor import cprint
import json
import numpy as np
import cv2
from pathlib import Path
#from video_play.play_window import play_window
from smooth_dectection import  smooth_dectection,Threshold_determination

posation_dist=[]
pool_posation=[]
pool_weight=0
pool_number=0
color_th=3
#用于记录状态
have_value=0
state_list=["finding region","unfull","full"]
pool_height=0
liquit_height=0
state=0#未定位-1，范围，
#已定位0 但是未满
#已定位1，已满
global posation_dist_label
posation_dist_label=0
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(10)
    # cv2.destroyAllWindows()
# if __name__ == "__main__":
#     # code to read image
#     image = cv.imread("D:\graduation\cooling_pool_emation/20220405213128.png", 0)
#     cv.imshow("Original", image)
#     smoothed_image = smooth(image)
#     cv.imshow("GaussinSmooth(5*5)", smoothed_image)
#     gradients, direction = get_gradient_and_direction(smoothed_image)
#     # print(gradients)
#     # print(direction)
#     nms = NMS(gradients, direction)
#     output_image = double_threshold(nms, 40, 100)
#     cv.imshow("outputImage", output_image)
#     cv.waitKey(0)
def bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2):
    '''
    说明：图像中，从左往右是 x 轴（0~无穷大），从上往下是 y 轴（0~无穷大），从左往右是宽度 w ，从上往下是高度 h
    :param x1: 第一个框的左上角 x 坐标
    :param y1: 第一个框的左上角 y 坐标
    :param w1: 第一幅图中的检测框的宽度
    :param h1: 第一幅图中的检测框的高度
    :param x2: 第二个框的左上角 x 坐标
    :param y2:
    :param w2:
    :param h2:
    :return: 两个如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
    '''
    # cv2.rectangle(draw_img, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 0), 8)
    # cv2.rectangle(draw_img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 10)
    if(x1>x2+w2):
        print("x1>x2+w2",x1,x2+w2)
        return 0
    if(y1>y2+h2):
        print("2")
        return 0
    if(x1+w1<x2):
        print("3")
        return 0
    if(y1+h1<y2):
        print("4")
        return 0
    colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    if(overlap_area==0):
        return 1
    return overlap_area / area2

def complete_similarity(x, y, w, h, con,frame_area,draw_img):
#选取面积增大的范围
#如果面积没有增大
    con_area=cv2.contourArea(con)
    samilarity=0
    unsamilarity=0
    for i in range(len(posation_dist)):
        print("i",i)
        similarity_number= 0
        l_x=posation_dist[i]["x"]
        l_y=posation_dist[i]["y"]
        l_h=posation_dist[i]["h"]
        l_w=posation_dist[i]["w"]
        #print(x,y)
        #print(posation_inf[0])
        m=posation_dist[i]["random_pos"][0]
        n=posation_dist[i]["random_pos"][1]
        if posation_dist[i]["area"]<=con_area:
            samples = sample(x, y, h, w, con)
            #draw_img = cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            #print("m",m,"n",n)
            for j in range(len(m)):
                a=int(l_x + m[j])
                b=int(l_y + n[j])
                if cv2.pointPolygonTest(con, (a,b), False) != -1:
                    similarity_number+=1

            if float(similarity_number)/len(m)>=con_area/frame_area:
                #最大范围
                posation_dist[i]["x"]=x if x<l_x else l_x
                posation_dist[i]["y"] =y if y<l_y else l_y
                posation_dist[i]["h"] = h if h>l_h else l_h
                posation_dist[i]["w"]=w if w>l_w else l_w
                posation_dist[i]['random_pos']=samples['random_pos']
                samilarity=1
                posation_dist[i]['similarity']+=1
            else:
                #可能是同一位置色块只是变形很大
                #判断中心点距离
                l_centerx=l_x+l_w/2
                l_centery = l_y + l_h / 2
                centerx=x+w/2
                centery=y+h/2

                p4 = abs(l_centery-centery)
                #
                print(bb_overlab(l_x, l_y, l_w, l_h, x, y, w, h))
                if p4<l_h/8 or bb_overlab(l_x, l_y, l_w, l_h, x, y, w, h)>0.8:
                    #如果中心点距离小于1/2宽，认为是同一色块
                    posation_dist[i]["x"] = x if x < l_x else l_x
                    posation_dist[i]["y"] = y if y < l_y else l_y
                    posation_dist[i]["h"] = h if h > l_h else l_h
                    posation_dist[i]["w"] = w if w > l_w else l_w
                    posation_dist[i]['random_pos'] = samples['random_pos']
                    samilarity = 1
                    posation_dist[i]['similarity'] += 1
                    continue

                else:
                    posation_dist[i]['unsimilarity'] += 1
                #     #如何去除干扰部分

        else:        #posation_dist.append(sample(x, y, h, w, con))
            l_centerx = l_x + l_w / 2
            l_centery = l_y + l_h / 2
            centerx = x + w / 2
            centery = y + h / 2

            p4 = abs(l_centery - centery)
            samples = sample(x, y, h, w, con)
            #draw_img = cv2.rectangle(draw_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            #当前色块面积小，但是是同一色块
            m = posation_dist[i]["random_pos"][0]
            n = posation_dist[i]["random_pos"][1]
            for j in range(len(m)):

                a = int(l_x + m[j])
                b = int(l_y + n[j])
                if cv2.pointPolygonTest(con, (a, b), False) != -1:
                    similarity_number += 1
            print("similarity", float(similarity_number) / len(m))

            if float(similarity_number) / len(m) >= con_area / frame_area :
                #最大范围
                #draw_img = cv2.rectangle(draw_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                posation_dist[i]["x"]=x if x<l_x else l_x
                posation_dist[i]["y"] =y if y<l_y else l_y
                posation_dist[i]["h"] = h if h>l_h else l_h
                posation_dist[i]["w"]=w if w>l_w else l_w

                #posation_dist[i]['random_pos']=samples['random_pos']
                samilarity=1
                posation_dist[i]['similarity'] += 1
            else:
                if(bb_overlab(l_x, l_y, l_w, l_h, x, y, w, h) > 0.8)  :
                    posation_dist[i]["x"] = x if x < l_x else l_x
                    posation_dist[i]["y"] = y if y < l_y else l_y
                    posation_dist[i]["h"] = h if h > l_h else l_h
                    posation_dist[i]["w"] = w if w > l_w else l_w

                    #posation_dist[i]['random_pos'] = samples['random_pos']
                    samilarity = 1
                    posation_dist[i]['similarity'] += 1
                else:
                    posation_dist[i]['unsimilarity'] += 1
    print("samilarity",samilarity)
    if samilarity!=0:
        return 1
    if samilarity==0 and posation_dist_label==0:
        posation_dist.append(sample(x,y,h,w,con))


def range_similarity(x, y, w, h,id):
#选取面积增大的范围
#如果面积没有增大
    samilarity=0
    unsamilarity=0
    for i in range(len(posation_dist)):
        if i==id:
            continue
        similarity_number= 0
        l_x=posation_dist[i]["x"]
        l_y=posation_dist[i]["y"]
        l_h=posation_dist[i]["h"]
        l_w=posation_dist[i]["w"]
        l_centery = l_y + l_h / 2

        centery=y+h/2

        p4 = abs(l_centery-centery)
        #
        print(bb_overlab(l_x, l_y, l_w, l_h, x, y, w, h))
        if p4<l_h/8 or bb_overlab(l_x, l_y, l_w, l_h, x, y, w, h)>0.8:
            #如果中心点距离小于1/2宽，认为是同一色块
            posation_dist[i]["x"] = x if x < l_x else l_x
            posation_dist[i]["y"] = y if y < l_y else l_y
            posation_dist[i]["h"] = h if h > l_h else l_h
            posation_dist[i]["w"] = w if w > l_w else l_w
            return -1
    return 0

def sample(x,y,h,w,con):
    # dist = cv2.pointPolygonTest(i,(50, 50),False)
    sample_mat=np.full((h, w),0)
    #print(h,w)
    for i  in range(0, h, 10):
        for j in range(0, w, 10):
            #print(j)
            if cv2.pointPolygonTest(con,(x+i ,y+j), False)!=-1:
                sample_mat[i][j]=1

    m, n = np.where(sample_mat>0)
    #print(sample_mat[m[1],n[1]])
    i = np.random.randint(len(m),size=int(len(m)/5))
    #print(i)
    random_pos = [m[i], n[i]]
    #print(random_pos[0])
    random_posation = {
        'random_pos': np.array(random_pos), 'x': x,'y':y,'h':h,'w':w,"area":cv2.contourArea(con),"similarity":0,"unsimilarity":0
    }
    return random_posation



#定义一个识别目标颜色并处理的函数
def select_color_img(target_color,erode_hsv,color_dist):
        for i in target_color:
            #print(color_dist[i]['Upper'].dtype)
            mask=cv2.inRange(erode_hsv,color_dist[i]['Lower'],color_dist[i]['Upper'])
            if(i==target_color[0]):
                inRange_hsv=cv2.bitwise_and(erode_hsv,erode_hsv,mask = mask)
            else:
                inRange_hsv1=cv2.bitwise_and(erode_hsv,erode_hsv,mask = mask)
                inRange_hsv=cv2.add(inRange_hsv,inRange_hsv1)
        return  inRange_hsv

#定义一个提取轮廓的函数
def extract_contour(final_inRange_hsv):
    inRange_gray = cv2.cvtColor(final_inRange_hsv,cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(inRange_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    return contours
#定义一个寻找目标并绘制外接矩形的函数
def find_target(contours,draw_img,target_list):
    con=None
    frame_shape=draw_img.shape
    frame_area=frame_shape[0]*frame_shape[1]
    for c in contours:
        if cv2.contourArea(c) < 3000:             #过滤掉较面积小的物体
            continue
        else:
            target_list.append(c)               #将面积较大的物体视为目标并存入目标列表
    for i in target_list:                       #绘制目标外接矩形
        x, y, w, h = cv2.boundingRect(i)
        cv2.drawContours(draw_img,i,-1,(0,255,0),5)
        # print("输入",x, y, w, h)
        if(len(posation_dist)==0):
            samples=sample(x, y, h, w, i)
            posation_dist.append(samples)
        similarity=complete_similarity(x, y, w, h, i,frame_area,draw_img)
        if similarity==1:
            con=i
        draw_img = cv2.rectangle(draw_img, (x, y), (x + w, y + h), (255, 255, 0), 3)
    return draw_img,con
#定义一个绘制中心点坐标的函数
def find_pool(draw_img,center):
    shape = draw_img.shape
    number=len(posation_dist)
    print("range_number",len(posation_dist))
    i=0
    delete_number=0
    for j in range (number):
        i+=1
        if i==number-delete_number:
            break
        s=posation_dist[i]
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]


        if range_similarity(x,y,w,h,i)==-1:
            #删除当前对象
            posation_dist.pop(i)
            i+=-1
            delete_number+=1
    i=0


    print("range_number",len(posation_dist))
    number=len(posation_dist)
    relearn=0
    for j in range(number):
        if (len(posation_dist) == i):
            #到达末尾
            break
        s = posation_dist[i]
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        # cv2.rectangle(draw_img, (x, y), (x + w, y + h), (100, 0, 0), 3)
        similarity=s["similarity"]
        unsimilarity=s["unsimilarity"]
        print("similarity,unsimilarity",similarity,unsimilarity)
        #去除干扰明显部分
        if unsimilarity-similarity>10 :
            #去除干扰
            posation_dist.pop(i)
            continue

        if similarity-unsimilarity>10:
            #但是其他部分相似值大于本
            # 重新学习区域
            if len(posation_dist)!=1:
                k=len(posation_dist)
                tmp_z=0
                tmp_label="n"
                for z in range(k):
                    if z == i:
                        continue
                    l_similarity = posation_dist[z]["similarity"]
                    l_unsimilarity= posation_dist[z]["unsimilarity"]
                    if l_similarity>similarity and l_similarity-l_unsimilarity>5:
                        posation_dist.clear()
                        relearn=1
                        break
            if relearn==1:
                break
            #标记为锅，并且不在添加其他区域
            global posation_dist_label
            posation_dist_label=1
            #去除非锅部分
            global pool_posation,state
            pool_posation=posation_dist[i]
            state=1
            i=0
            x = pool_posation["x"]
            w = pool_posation["w"]
            y = pool_posation["y"]
            h =pool_posation["h"]
            pool_posation["x"] = x -int( w / 4) if x -int( w / 4) > 0 else 0
            pool_posation["w"] = w + int( w / 4) if x - int( w / 4) > 0 else w + x
            pool_posation["y"] = y - 20 if y-20 > 0 else 0
            pool_posation["h"] = h + 20 if y- 20 > 0 else h+y
            print("标记")

        i += 1
    cv2.imshow("range",draw_img)
    return draw_img

def detection_state(draw_img):
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    x = pool_posation["x"]
    w = pool_posation["w"]
    y = pool_posation["y"]
    h = pool_posation["h"]
    cropped = draw_img[y:y + h, x:x + w]
    data = cropped.reshape((-1, 3))
    data = np.float32(data)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels4, centers4 = cv2.kmeans(data, 3, None, criteria, 10, flags)
    centers4 = np.uint8(centers4)
    res = centers4[labels4.flatten()]
    dst4 = res.reshape((cropped.shape))
    # 保存截取的图片
    x = centers4[0]
    tmp = int(x[0]) + int(x[1]) + int(x[2])
    tmp_i = 0

    for z in range(len(centers4)):
        # print("x",x)
        # 图像数值超出
        x = centers4[z]
        a = int(x[0]) + int(x[1]) + int(x[2])
        if a <= tmp:
            tmp = a
            tmp_i = x
    print(tmp_i)
    mask = cv2.inRange(dst4, np.array([0, 0, 0]), np.array([int(tmp_i[0]), int(tmp_i[1]), int(tmp_i[2])]))
    # image = cv2.imread("D:\graduation\cooling_pool_emation/20220405213128.png", 0)
    # cv2.imshow("Original", image)
    # image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # smoothed_image = smooth(image)
    # #cv2.imshow("GaussinSmooth(5*5)", smoothed_image)
    # gradients, direction = get_gradient_and_direction(smoothed_image)
    # # print(gradients)
    # # print(direction)
    # nms = NMS(gradients, direction)
    # output_image = double_threshold(nms, 40, 100)
    # cropped=cv2.cvtColor(cropped, cv2.COLOR_BGR2HLS)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(cropped, (5, 5), 4)
    image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
    mean = int(image.mean())
    _, first = cv2.threshold(image, maxVal - 4 * int(maxVal - mean) / 11, 255, cv2.THRESH_BINARY)
    first = cv2.cvtColor(first, cv2.COLOR_GRAY2RGB)

    # result = cv2.add(dst4, first)
    result = cv2.add(mask, first)
    # img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
    #
    # img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("Original", result)
    # mean = int(iresult.mean())
    # _, result = cv2.threshold(img, mean, 255, cv2.THRESH_BINARY)

    # result=cv2.medianBlur(groud, 9)

    result = cv2.medianBlur(result, 9)
    cv2.imshow("reslut", dst4)
    edges = cv2.Canny(result, 100, 200, L2gradient=True)
    # edges2 = cv2.Canny(cropped, 50, 200)
    # output_image = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    # plt.subplot(131), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # cv2.imshow("outputImage", output_image)
    cv2.imshow("cut", result)
    cv2.imshow("edges", edges)
    # cv2.circle(draw_img, (center_x, center_y), 3, 70, -1)  # 绘制中心点
    # str1 = '(' + str(center_x) + ',' + str(center_y) + ')'  # 把坐标转化为字符串
    # cv2.putText(draw_img, str1, (center_x - 50, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
    #             cv2.LINE_AA)  # 绘制坐标点位





def draw_center(draw_img,center):
    shape = draw_img.shape
    number=len(posation_dist)
    print("range_number",len(posation_dist))
    i=0
    delete_number=0
    for j in range (number):
        i+=1
        if i==number-delete_number:
            break
        s=posation_dist[i]
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        if range_similarity(x,y,w,h,i)==-1:
            #删除当前对象
            posation_dist.pop(i)
            i+=-1
            delete_number+=1
    i=0
    print("range_number",len(posation_dist))
    number=len(posation_dist)
    relearn=0
    for j in range(number):
        if (len(posation_dist) == i):
            #到达末尾
            break
        s = posation_dist[i]

        similarity=s["similarity"]
        unsimilarity=s["unsimilarity"]
        print("similarity,unsimilarity",similarity,unsimilarity)
        #去除干扰明显部分
        if unsimilarity-similarity>10 :
            #去除干扰
            posation_dist.pop(i)
            continue

        if similarity-unsimilarity>50:
            #但是其他部分相似值大于本
            # 重新学习区域
            if len(posation_dist)!=1:
                k=len(posation_dist)
                tmp_z=0
                tmp_label="n"
                for z in range(k):
                    if z == i:
                        continue
                    l_similarity = posation_dist[z]["similarity"]
                    l_unsimilarity= posation_dist[z]["unsimilarity"]
                    if l_similarity>similarity and l_similarity-l_unsimilarity>10:
                        posation_dist.clear()

                        relearn=1
                        break




            if relearn==1:
                break
            #标记为锅，并且不在添加其他区域
            global posation_dist_label
            posation_dist_label=1
            #去除非锅部分
            if len(posation_dist)!=1:
                #还存在其他预估部分
                for m in range(len(posation_dist)):
                    if m!=i:
                        posation_dist.pop(m)
            i=0
            x = posation_dist[0]["x"]
            w = posation_dist[0]["w"]
            y = posation_dist[0]["y"]
            h = posation_dist[0]["h"]
            posation_dist[0]["x"] = x -int( w / 4) if x -int( w / 4) > 0 else 0
            posation_dist[0]["w"] = w + int( w / 4) if x - int( w / 4) > 0 else w + x
            posation_dist[0]["y"] = y - 20 if y-20 > 0 else 0
            posation_dist[0]["h"] = h + 20 if y- 20 > 0 else h+y
            print("标记")
        if relearn==1:
            break
        s=posation_dist[i]
        x, y, w, h = s["x"],s["y"],s["w"],s["h"]
        print(x, y, w, h)# 计算中心点的x、y坐标
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        width=shape[1]
        height=shape[0]
        con_x=0
        con_y=0
        # print('center_x:', center_x)  # 打印（返回）中心点的x、y坐标
        # print('center_y:', center_y)
        # cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 0), 3)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # 如果在已知范围的左上方，则扩大右下方范围
        print("center is not None:",center is not None)
        if center is not None:

            c = max(center, key=cv2.contourArea)
            # 确定面积最大的轮廓的外接圆
            #((con_x, con_y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(center)
            con_x = int(M['m10'] / M['m00'])  # 重心的x坐标
            con_y = int(M['m01'] / M['m00'])  # 重心的y坐标
            # print("con_x, con_y",con_x, con_y )
            # print("width,height",width,height)
            cv2.circle(draw_img, (int(con_x), int(con_y)), 5, 70, -1)
            #cv2.circle(draw_img, (int(con_x), int(con_y)), int(radius), (0, 0, 213), -1)
            # 计算轮廓的矩
            #计算可取最大范围
            tmp_x=x+(center_x-con_x)
            tmp_w=w+abs(center_x-con_x)
            tmp_y=y +(center_y - con_y)
            tmp_h=h+abs(center_y-con_y)
            # print("center_x,center_y,w,h", center_x,center_y,w,h)
            # print("tmp_x,tmp_x + tmp_w,tmp_y,tmp_y + tmp_h",tmp_x,tmp_x + tmp_w,tmp_y,tmp_y + tmp_h)
            if tmp_x < 0 or tmp_x + tmp_w > width or tmp_y < 0 or tmp_y + tmp_h> height:
                #超出
                print("超出")
            else:
                x = tmp_x
                w = tmp_w
                y = tmp_y
                h = tmp_h

        cropped = draw_img[y:y+h, x:x+w]
        data = cropped.reshape((-1, 3))
        data = np.float32(data)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels4, centers4 = cv2.kmeans(data, 3, None, criteria, 10, flags)
        centers4 = np.uint8(centers4)
        res = centers4[labels4.flatten()]
        dst4 = res.reshape((cropped.shape))
        # 保存截取的图片
        x = centers4[0]
        tmp = int(x[0]) + int(x[1]) + int(x[2])
        tmp_i = 0

        for z in range(len(centers4)):
            # print("x",x)
            # 图像数值超出
            x = centers4[z]
            a = int(x[0]) + int(x[1]) + int(x[2])
            if a <= tmp:
                tmp = a
                tmp_i = x
        print(tmp_i)
        mask = cv2.inRange(dst4, np.array([0, 0, 0]), np.array([int(tmp_i[0]), int(tmp_i[1]), int(tmp_i[2])]))
        #image = cv2.imread("D:\graduation\cooling_pool_emation/20220405213128.png", 0)
        # cv2.imshow("Original", image)
        # image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # smoothed_image = smooth(image)
        # #cv2.imshow("GaussinSmooth(5*5)", smoothed_image)
        # gradients, direction = get_gradient_and_direction(smoothed_image)
        # # print(gradients)
        # # print(direction)
        # nms = NMS(gradients, direction)
        # output_image = double_threshold(nms, 40, 100)
        #cropped=cv2.cvtColor(cropped, cv2.COLOR_BGR2HLS)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blurred = cv2.GaussianBlur(cropped, (5, 5), 4)
        image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
        mean = int(image.mean())
        _, first = cv2.threshold(image, maxVal - 4 * int(maxVal - mean) / 11, 255, cv2.THRESH_BINARY)
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2RGB)

        # result = cv2.add(dst4, first)
        result = cv2.add(mask, first)
        # img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
        #
        # img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("Original", result)
        # mean = int(iresult.mean())
        # _, result = cv2.threshold(img, mean, 255, cv2.THRESH_BINARY)



        # result=cv2.medianBlur(groud, 9)

        result = cv2.medianBlur( result, 9)
        cv2.imshow("reslut", dst4)
        edges = cv2.Canny(result, 100, 200,L2gradient=True)
        # edges2 = cv2.Canny(cropped, 50, 200)
        # output_image = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        # plt.subplot(131), plt.imshow(img, cmap='gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        #cv2.imshow("outputImage", output_image)
        cv2.imshow("cut", result)
        cv2.imshow("edges",edges)
        #cv2.circle(draw_img, (center_x, center_y), 3, 70, -1)  # 绘制中心点
        # str1 = '(' + str(center_x) + ',' + str(center_y) + ')'  # 把坐标转化为字符串
        # cv2.putText(draw_img, str1, (center_x - 50, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
        #             cv2.LINE_AA)  # 绘制坐标点位
        i+=1
    return draw_img

class MyThread:

    def __init__(self, color='g',fileName='',savepath='',timeRate=0.05):
        self.color = color
        self.t_n = 0  ## 总线程数
        self.f_n = 0  ## 已完成线程数
        self.startTime = time.time()  ## 开始时间
        self.q = queue.Queue()  ## 收集容器
        self.pro = True  ## 是否打印进度条
        self.frame=None #当前视频帧
        self.filename = fileName
        self.cap=cv2.VideoCapture(self.filename) #当前视频流
        self.timeRate=timeRate
        self.savepath=savepath
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.ret=None
    def progressBar(self, f_n, t_n):  ## 进度条
        if not self.pro:
            return
        if self.f_n < self.t_n:
            end = ""
        else:
            self.pro = False
            end = "\n"

        n = f_n / t_n  ## 计算完成率

        elapsedTime = datetime.timedelta(seconds=int(time.time() - self.startTime))
        sta_s = int(time.time() - self.startTime) / f_n * (t_n - f_n)  ## 已用时间
        eta = datetime.timedelta(seconds=int(sta_s))  ## 还需时间估计

        cprint("\r" + "({} of {})".format(f_n, t_n).ljust(2 * len(str(t_n)) + 6) + " [" +
               ("=" * int(100 * n / 2) + ">").ljust(50) + "] {}%".format(round(n * 100, 2)).ljust(8) +
               "  Elapsed Time: {}  ETA: {}".format(elapsedTime, eta) + end, color='green', end='')

    def splitList(self, n, dataList):
        """
        将一个列表分成 n 份
        :param n:
        :param dataList:
        :return:
        """
        per_th = math.ceil(len(dataList) / n)
        res = []
        for i in range(n):
            res.append(dataList[per_th * i:per_th * (i + 1)])
        return res

    def video_target(self,):
        #self.cap = cv2.VideoCapture(self.filename)  # 导入的视频所在路径
        start_time = time.time()
        counter = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # 视频平均帧率
        width, height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        picture_number=1
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
        out = cv2.VideoWriter('test2.mp4', fourcc, self.fps, (width, height))
        kernel = np.ones((5, 5), np.uint8)
        kernel1 = np.ones((10, 10), np.uint8)
        while self.cap.isOpened():
            print("视频流打开成功")
            self.ret, self.frame = self.cap.read()

            #print("self.ret",self.ret," self.frame", self.frame)
            # 键盘输入空格暂停，输入q退出
            key = cv2.waitKey(1) & 0xff
            if key == ord(" "):
                cv2.waitKey(0)
            if key == ord("q"):
                break

            counter += 1  # 计算帧数
            # if (time.time() - start_time) != 0:  # 实时显示帧数
            #     cv2.putText(self.frame, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (500, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
            #                 3)

            #     # print("FPS: ", counter / (time.time() - start_time))
            #     counter = 0
            #     start_time = time.time()
            if self.ret:
                global  state_list,state,have_value,pool_height,liquit_height
                frame=self.frame.copy()
                frame=cv2.putText(frame, state_list[state], (0,50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (200, 200, 200), 5)
                if have_value==1:
                    text1="pool_height="+str(pool_height)
                    text2="liquit_height="+str(liquit_height)
                    frame =cv2.putText(frame,text1, (0, 80), cv2.FONT_HERSHEY_COMPLEX, 1.0, (200, 200, 200),  3)
                    frame = cv2.putText(frame, text2, (0, 120), cv2.FONT_HERSHEY_COMPLEX, 1.0, (200, 200, 200), 3)
                if state!=0:
                    x = pool_posation["x"]
                    w = pool_posation["w"]
                    y = pool_posation["y"]
                    h = pool_posation["h"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 20, 100), 3)
                cv2.imshow('frame', frame)
                out.write(frame)  # 写入帧
                frameRate = int(self.cap.get(5)) * self.timeRate  # 因为cap.get(5)获取的帧数不是整数，所以需要取整一下（向下取整用int，四舍五入用round，向上取整需要用math模块的ceil()方法）
                if (picture_number % frameRate == 0):
                    have_value=0
                    print("开始截取视频第：" + str(picture_number) + " 帧")
                    # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
                    #cv2.imshow("cut", self.frame)
                    # data = self.frame.reshape((-1, 3))
                    # data = np.float32(data)
                    # flags = cv2.KMEANS_RANDOM_CENTERS
                    # compactness, labels4, centers4 = cv2.kmeans(data, 7, None, criteria, 10, flags)
                    # centers4 = np.uint8(centers4)
                    # res = centers4[labels4.flatten()]
                    # dst4 = res.reshape((self.frame.shape))
                    #cv2.imshow("kmeans", dst4)
                    #dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
                    # print("label4",labels4)
                    # print("centers4",centers4)
                    # print("res", res)
                    #cv2.imshow("kmeans",dst4)

                    if posation_dist_label!=1:

                        # data1=cv2.cvtColor(self.frame, cv2.COLOR_BGR2HLS)
                        # cv2.imshow("hls",data1)

                        data=self.frame.copy()
                        data = cv2.erode(data, kernel, iterations=1)
                        data1 = cv2.dilate(data, kernel1, iterations=3)
                        data1 = cv2.GaussianBlur(data1, (5, 5), 4)
                        data = data1.reshape((-1, 3))
                        data = np.float32(data)
                        flags = cv2.KMEANS_RANDOM_CENTERS
                        compactness, labels4, centers4 = cv2.kmeans(data, 7, None, criteria, 10, flags)
                        centers4 = np.uint8(centers4)
                        res = centers4[labels4.flatten()]
                        dst4 = res.reshape((self.frame.shape))
                        tmp=0
                        tmp_i=0
                        for x in centers4:
                            # print("x",x)
                            # 图像数值超出
                            a = int(x[0]) + int(x[1]) + int(x[2])
                            if a >=tmp:
                                tmp=a
                                tmp_i=x

                            # print(x[0], x[1], x[2])
                            # print("a", a)
                        color_dist = {
                            'yellow': {'Lower': np.array([int(tmp_i[0]),int(tmp_i[1]),int(tmp_i[2])]), 'Upper': np.array([255, 255, 255])},
                        }
                        # 目标颜色
                        target_color = ['yellow']
                        # 创建目标列表
                        target_list = []
                        #img = cv2.imread(self.frame, cv2.COLOR_BGR2RGB)  # 读入图像（直接读入灰度图）
                        draw_img = data1.copy()
                        draw_img0=draw_img# 为保护原图像不被更改而copy了一份，下面对图像的修改都是对这个副本进行的
                        # cv2.erode(draw_img, draw_img, se)
                        #
                        # cv2.dilate(dst, dst, se)
                        data2=self.frame.copy()

                        final_inRange_hsv = select_color_img(target_color,data1,color_dist)
                        contours = extract_contour(final_inRange_hsv)
                        draw_img ,center= find_target(contours, data1,target_list)
                        final_img = find_pool(draw_img ,center)
                        cv_show('final_img', final_img)
                    else:
                        state=1
                        draw_img=self.frame.copy()
                        center = None
                        x = pool_posation["x"]
                        w = pool_posation["w"]
                        y=pool_posation["y"]
                        h=pool_posation["h"]
                        #仅分割已固定范围
                        cropped = draw_img[y:y + h, x:x + w]
                        # data = cropped.reshape((-1, 3))
                        # data = np.float32(data)
                        # flags = cv2.KMEANS_RANDOM_CENTERS
                        # compactness, labels4, centers4 = cv2.kmeans(data, 7, None, criteria, 10, flags)
                        # centers4 = np.uint8(centers4)
                        # res = centers4[labels4.flatten()]
                        # dst4 = res.reshape((cropped.shape))
                        # 保存截取的图片

                        # image = cv2.imread("D:\graduation\cooling_pool_emation/20220405213128.png", 0)
                        # cv2.imshow("Original", image)

                        # cropped=cv2.cvtColor(cropped, cv2.COLOR_RGB2HLS)
                        data = cropped.reshape((-1, 3))
                        data = np.float32(data)
                        flags = cv2.KMEANS_RANDOM_CENTERS
                        compactness, labels4, centers4 = cv2.kmeans(data, 3, None, criteria, 10, flags)
                        centers4 = np.uint8(centers4)
                        res = centers4[labels4.flatten()]
                        dst4 = res.reshape((cropped.shape))
                        x = centers4[0]
                        tmp = int(x[0]) + int(x[1]) + int(x[2])
                        tmp_i = 0
                        for z in range(len(centers4)):
                            # print("x",x)
                            # 图像数值超出
                            x = centers4[z]
                            a = int(x[0]) + int(x[1]) + int(x[2])
                            if a <= tmp:
                                tmp = a
                                tmp_i = x
                        print(tmp_i)
                        mask = cv2.inRange(dst4, np.array([0, 0, 0]),
                                           np.array([int(tmp_i[0]), int(tmp_i[1]), int(tmp_i[2])]))
                        mask = np.array(mask)
                        mask[mask == 255] = 125
                        pool=mask
                        # image = cv2.imread("D:\graduation\cooling_pool_emation/20220405213128.png", 0)
                        # cv2.imshow("Original", image)
                        # image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        # smoothed_image = smooth(image)
                        # #cv2.imshow("GaussinSmooth(5*5)", smoothed_image)
                        # gradients, direction = get_gradient_and_direction(smoothed_image)
                        # # print(gradients)
                        # # print(direction)
                        # nms = NMS(gradients, direction)
                        # output_image = double_threshold(nms, 40, 100)
                        # cropped=cv2.cvtColor(cropped, cv2.COLOR_BGR2HLS)
                        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        # image = cv2.cvtColor(dst4, cv2.COLOR_BGR2GRAY)
                        # smoothed_image = smooth(image)
                        # # cv2.imshow("GaussinSmooth(5*5)", smoothed_image)
                        # gradients, direction = get_gradient_and_direction(smoothed_image)
                        # # print(gradients)
                        # # print(direction)
                        # nms = NMS(gradients, direction)
                        # output_image = double_threshold(nms, 40, 100)

                        # cv2.imshow("outputImage", output_image)
                        #dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2HLS)
                        cv2.imshow("mask", pool)


                        blurred = cv2.GaussianBlur(cropped, (5, 5), 4)
                        image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
                        mean = int(image.mean())
                        global  color_th
                        _, groud = cv2.threshold(image, maxVal - 6* int(maxVal - mean) / 11, 255, cv2.THRESH_BINARY)
                        groud = cv2.cvtColor(groud, cv2.COLOR_GRAY2RGB)
                        res2 = cv2.add(groud, mask)

                        cv2.imshow("cut",dst4)
                        #将截取的图像保存在本地
                        res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
                        xy = np.column_stack(np.where(pool == 0))
                        # y轴在前，x轴在后
                        plt.figure(1)
                        n1, bins1, patches1 = plt.hist(xy[:, 1], bins=int(cropped.shape[1] / 20), range=(0,cropped.shape[1]),facecolor="blue",
                                                       edgecolor="black", alpha=0.7)

                        print(len(n1), len(bins1), patches1)

                        plt.xlabel("x")
                        # 显示纵轴标签
                        plt.ylabel("black")
                        plt.figure(2)
                        xy = np.column_stack(np.where(res2 == 255))
                        n2, bins2, patches2 = plt.hist(xy[:, 1], bins=int(cropped.shape[1] / 20),range=(0,cropped.shape[1]), facecolor="blue",
                                                       edgecolor="black", alpha=0.7)
                        print(len(n2), bins2, patches2)

                        # 获取较为平滑的一段
                        # for i in n1:
                        #     # 去除无液面区域
                        #     if n2<1000:
                        # 去除白色面积大于黑色部分
                        #
                        cv2.imshow("result3", res2)
                        # 按列读取并且取出灰色与白色之间部分
                        # 图片格式：y行x列
                        # 0到x列
                        xy = [[0, 0]]
                        white_xy = [[0, 0]]
                        for x in range(res2.shape[1]):
                            tmp_xy = []
                            tmp_white_xy = []
                            real_number = 0
                            black_number = 0
                            white_number = 0
                            i = 0

                            # print(res2[:,0])
                            for y in range(res2.shape[0]):

                                if res2[y][x] == 0:
                                    # 为黑色
                                    if white_number != 0:
                                        white_xy = np.concatenate((white_xy, tmp_white_xy), axis=0)
                                        break
                                    black_number += 1
                                    tmp_xy.append([y, x])
                                elif res2[y][x] == 255:
                                    # 为白色
                                    white_number += 1
                                    tmp_white_xy.append([y, x])
                                    if (black_number != 0 and real_number != 1):
                                        # print("tmp_xy",tmp_xy)
                                        real_number = 1
                                        xy = np.concatenate((xy, tmp_xy), axis=0)
                                i += 1

                        plt.figure(3)
                        if len(xy)==1:
                            picture_number += 1
                            continue
                        n3, bins3, patches3 = plt.hist(xy[:, 1], bins=int(cropped.shape[1] / 20), range=(0,cropped.shape[1]),facecolor="blue",
                                                       edgecolor="black", alpha=0.7)

                        #液体部分大于黑色距离
                        for white_number_i in range(len(n2)):
                            print(n2[white_number_i]/n1[white_number_i])
                            if n2[white_number_i]/n1[white_number_i]>4/5:
                                n3[white_number_i] = 0

                        plt.plot(bins3[1:], n3, 'green')
                        # plt.show()
                        plt.figure(4)
                        n4, bins4, patches4 = plt.hist(white_xy[:, 1], bins=int(cropped.shape[1] / 20), facecolor="blue",range=(0,cropped.shape[1]),
                                                       edgecolor="black", alpha=0.7)

                        threshold=Threshold_determination(n3,bins3,n4,bins4)
                        # if threshold > 50:
                        #     picture_number+=1
                        #     continue
                        print("threshold", threshold)
                        smooth_x, avg_value, smooth_y = smooth_dectection(bins3, n3, threshold*2)
                        # smooth_white_x, avg_white_value, smooth_white_y = smooth_dectection(bins4, n4, threshold*2)
                        # if len(smooth_x)==0 or len(smooth_white_x)==0 :
                        #     picture_number+=1
                        #     print("无平缓")
                        #     continue
                        #
                        # x_start = 0
                        # x_end = 0
                        #
                        # if smooth_x[1] < smooth_white_x[0] or smooth_x[0] > smooth_white_x[1]:
                        #     print("不相交")
                        # else:
                        #     if smooth_x[0] < smooth_white_x[0]:
                        #         x_start = smooth_white_x[0]
                        #     else:
                        #         x_start = smooth_x[0]
                        #     if smooth_x[1] < smooth_white_x[1]:
                        #         x_end = smooth_x[1]
                        #     else:
                        #         x_end = smooth_white_x[1]

                        if len(smooth_x) == 0 :
                                picture_number+=1
                                print("无平缓")
                                continue

                        x_start = smooth_x[0]
                        x_end = smooth_x[1]
                        if x_end-x_start<=3*cropped.shape[1]/20:
                            print("过短")
                            picture_number+=1
                            continue
                        print("avg", avg_value)

                        weight1_2 = (bins3[1] - bins3[0])
                        # for number in range (len(n3)):
                        result = res2[0:, int(x_start):int(x_end - weight1_2)]

                        cv2.rectangle(res2, (int(x_start), 0), (int(x_end - weight1_2), cropped.shape[0]), (40, 5, 245), 8)
                        # 已返回有用区域，则使用该部分的锅液距离估计状态
                        # 状态的估计方法：锅液距离与锅的宽度的关系
                        # 锅液距离<=可用区域锅宽度*1/3
                        cv2.imshow("smooth_region",res2)
                        i = -1
                        smooth_number = 0
                        sum_y = 0
                        pool_weight_tmp=0
                        for x in bins3:
                            i += 1
                            if x >= x_start and x < x_end:

                                pool_weight_tmp+=n1[i]
                                sum_y += (n4[i]+n3[i])
                                smooth_number += 1

                            else:
                                picture_number+=1
                                continue

                        avg_pool_wight = sum_y / smooth_number
                        global pool_weight,pool_number
                        have_value=1
                        pool_height=avg_pool_wight
                        liquit_height=avg_pool_wight-avg_value[0]
                        pool_weight_tmp=pool_weight_tmp/smooth_number
                        pool_number+=1
                        print("pool_number", pool_number)
                        pool_weight+=pool_weight_tmp
                        if pool_number==10:
                            pool_weight=(pool_weight)/pool_number
                            pool_weight=pool_weight/(cropped.shape[1] / 20)
                            print("pool",pool_weight)
                            y = pool_posation["y"]
                            h = pool_posation["h"]
                            if pool_weight+50<=h:
                                pool_posation["y"]=int(y+(h-pool_weight-50))
                                pool_posation["h"]=int(pool_weight+50)
                                color_th=5
                        print("avg_pool_wight", avg_pool_wight)
                        if avg_value[0] <= 12*avg_pool_wight / 26:
                            print("full")

                            state=2

                picture_number += 1

            else:
                out.release()
                print("所有帧都已经保存完成")
                break
            time.sleep(1 / self.fps)  # 按原帧率播放
        print("视频流播放完成")

    def change_color(self,frame_img,kmeans_img,centers4, arr=None):
        # 需要更换的像素值
        #arr=center4//四种分类
        scr_img = centers4
        #排序
#包含4个分类像素值
        scr_img.sort(key=lambda x: x[0]+x[1]+x[2])
        # 新的像素值
        new_img = [0, 0, 0]

        start_time = time.time()
        # 三通道分离
        r_img, g_img, b_img = scr_img[:, :, 0].copy(), scr_img[:, :, 1].copy(), scr_img[:, :, 2].copy()
        r_frame, g_frame, b_frame = frame_img[:, :, 0].copy(), frame_img[:, :, 1].copy(),frame_img[:, :, 2].copy()
        # 三通道值相加形式变成单通道，组合在一起，进行编码
        img = r_img+g_img+b_img
        value_const = kmeans_img[0] + kmeans_img[1] + kmeans_img[2]#kmeans 三通道的编码

        # 符合条件进行更改像素值
        r_frame[img[0] != value_const[0] and img[1] != value_const[0]] = new_img[0]
        g_frame[img[0] != value_const[0] and img[1] != value_const[0]] = new_img[1]
        b_frame[img[0] != value_const[0] and img[1] != value_const[0]] = new_img[2]

        # numpy三通道合并
        arr1 = np.dstack([r_frame, g_frame, b_frame])

        print('通道法用时时间：{}'.format(time.time() - start_time))

        plt.imsave('E:/2315.png', arr1)



    def image_target(self):
        time.sleep(1)
        picture_number = 1
         # 截取视频帧的时间间隔（这里是每隔10秒截取一帧）
        while (True):
            if self.ret:
                frameRate = int(self.fps) * self.timeRate  # 因为cap.get(5)获取的帧数不是整数，所以需要取整一下（向下取整用int，四舍五入用round，向上取整需要用math模块的ceil()方法）
                if (picture_number % frameRate == 0):
                    print("开始截取视频第：" + str(picture_number) + " 帧")
                    # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
                    #cv2.imshow("cut", self.frame)
                    print(self.frame)
                    #cv2.imwrite(self.filename +"/frame"+ str(picture_number) + '.jpg', self.frame)  # 这里是将截取的图像保存在本地
                picture_number += 1
                cv2.waitKey(0)
            else:
                print("所有帧都已经保存完成")
                break
        # release the video stream

    def mul_thread(self, timeout=2,color='g'):
        """
        多线程执行
        :param n: 同时开启的线程数
        :param fun: 函数
        :param argsList: 每个函数的参数， 列表形式
        :param timeout: 每个线程限制的时间
        :return: 以队列形式返回执行失败的参数
        """
        #self.__init__(self, color='g',fileName='',timeRate)
        # self.t_n = 2
        # self.color = color
        #
        # sema = threading.Semaphore(2)  ## 同时开启 n_t 个进程
        # threads = []
        # threads.append(Thread(target=self.video_target,args=()))
        # #time.sleep(1)
        # threads.append(Thread(target=self.image_target, args=()))
        # for thread in threads:  ## 开启所有进程
        #     thread.start()
        #     #time.sleep(0.05)
        # for thread in threads:  ## 等待所有线程完成
        #     thread.join(timeout)
        self.video_target()
        self.cap.release()
        cv2.destroyAllWindows()
        return self.q

    # def __call__(self, *args, **kwargs):
    #
    #     return self.mul_thread(*args, **kwargs)

if __name__ == '__main__':
    param = [
        ["D:\graduation\cooling_pool_emation\data/003.mp4", "D:\graduation\cooling_pool_emation\data/frame", 100],  # 视频路径；文件保存路径；按每100帧截取一帧，
    ]

    b=MyThread(color='g',fileName="D:\graduation\yolov5-master\dataset/val.mp4", savepath="D:\graduation\cooling_pool_emation\data/frame",timeRate=0.5)
    b.mul_thread()
