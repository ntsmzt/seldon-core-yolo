#!/usr/bin/env python
from __future__ import print_function


import copy
import argparse
import sys
import subprocess
import requests
import json
import time
import cv2
import numpy as np


def preprocess(orgimg):
        orgimg = orgimg.astype(np.uint8)
        if orgimg.shape[-1] == 4:
            orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGRA2BGR)
        im0 = copy.deepcopy(orgimg)
        imgsz = (1280, 1280)
        img = letterbox(im0, new_shape=imgsz)[0]
        return img

def letterbox(img, new_shape=(1280, 1280), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        iw, ih = int(dw / stride), int(dh / stride)
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh + ih * stride / 2 - 0.1)), int(round(dh + ih * stride / 2 + 0.1))
        left, right = int(round(dw + iw * stride / 2 - 0.1)), int(round(dw + iw * stride / 2 + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)




def save_image(num, image):
    """Save the images.

    Args:
        num: serial number
        image: image resource

    Returns:
        None
    """
    image_path = './raw_pictures/{}.jpg'.format(str(num))
    cv2.imwrite(image_path, image)


def send_image_array(frame_data):
    """Send image numpy array to aiplatform

    Args:
        num: serial number
        image: image resource

    Returns:
        None
    """

    print("============ frame_data type = ", frame_data.dtype)
    print("============================ in send_image_array =============")
    url = 'http://192.168.3.185:30164/seldon/seldon-app/demo-ccp-zt/api/v1.0/predictions'
    headers = {'content-type': 'application/json'}
    

    print("================frame_data", frame_data)   
    print("================frame_data size", frame_data.size)
    print("================frame_data shape", frame_data.shape)
    print("================frame_data ndim", frame_data.ndim)
    
    # ndarray = frame_data 
    ndarray = frame_data.flatten()
    # 24
    pic_lengh = 4915200
    s = 1

    ndarray = ndarray.reshape(s, int(pic_lengh/s))[0]
    

    print("================ndarray", ndarray)   
    print("================size", ndarray.size)
    print("================shape", ndarray.shape)
    print("================ndim", ndarray.ndim)
    print("================element type", type(ndarray[0]))
    


   
#    print("===============flatten time", end-start)


#    print("===============ndarray", ndarray)
	
#    sdarray = str(ndarray)   
#    res=sdarray.strip('[')
#    res=res.strip(']')
#    res=res.split(' ')     
    
   
#    print("===============res type", type(res))
#    print("===============res length", len(res))
#    print("===============res", res)
#    res = list(map(int, res))  

 
#    print("================sdarray", sdarray)
    # time.sleep(1)   
#    end = time.time()
#    print("===============to str time", end-start)
   

    start = time.time()	
    ndarray = ndarray.tolist()    	 
    end = time.time()
    print("===============to list time", (end-start)*s)
   

    send_start = time.time()
    # 以字典的形式构造数据
    request_data = {
        "data":{
            "names": 'camera01',
            "ndarray": ndarray
        }
    }
    
    
    #print("============================ request_data = ", request_data)
    # 与 get 请求一样，r 为响应对象
    r= requests.post(
            url, 
            json=request_data
        )
 
    send_end = time.time()
    print("================send time is :", send_end - send_start)
    #    r = requests.post(
    #        url,
    #        data=json.dumps(request_data),
    #        headers=headers
    #    )
    # 查看响应结果
 
    print(r.json())



    
if __name__ == '__main__':
    count = 0  # count the number of pictures
    frame_interval = 30  # video frame count interval frequency
    frame_interval_count = 0
    rtmp_url = "rtmp://192.168.3.186:1935/4k_live/R3_Ds_CamE_101_gpu03_2023-03-30-08-45-00"
    
    cap = cv2.VideoCapture(rtmp_url)
    if cap.isOpened():
    #读取视频中的帧
        ret,frame=cap.read()
    else:
        ret = False



    while ret:
        # Grab a single frame of video
        ret, frame = cap.read()
        print("==============ret==========", len(frame),len(frame[0]),len(frame[0][0]))
        print("==============read time==========", time.time())


        if(ret):
            count+=1;
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            fnum = int(cap.get(7))
            print("--------- w = ",w)
            print("--------- h = ",h)
            print("--------- fps = ",fps)
            print("--------- fnum = ",fnum)
            start = time.time()	
            small_frame = preprocess(frame)
            
            end = time.time()
            print("111111111111111111111",end-start)	
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            
            rgb_frame = small_frame[:, :, ::-1]

#            save_image(count, rgb_frame)
            
            send_image_array(rgb_frame)

 
            # Draw a box around the face
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            #cv2.waitKey(1) 
            # sleep()
        else:
            print("connection lost...")
            ret = False
            # video_capture = cv2.VideoCapture(rtmp_addr)

    # Release handle to the webcam
    video_capture.release()
    #cv2.destroyAllWindows()



