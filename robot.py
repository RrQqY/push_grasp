#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('my_pack')
import sim
import time
import cv2
import numpy as np


# 导入机器人正反解函数
class Robot():
    def __init__(self):
        self.clientID = -1
        self.move_step_speed = 0.05                 # 机器人移动步长
        
        self.moveL_step = 0.05
        self.range = [[0.25,0.75],[-0.25,0.25],[0.01,0.4]]    # 场地范围
        self.move_height = [0.05,0.4,0.05]       # 机械臂末端设定高度，0为抓取时下落，1为抓取时上升，2为推动时下落
        
        self.camera_handler = 0                  # 摄像头句柄
        self.arm_handler = 0                     # 机械臂句柄
        self.robot_pos_now = [0,0,0]             # 机器人当前位置
        self.robot_pos_aim = [0.5,0.1,0.3]       # 机器人目标位置
        self.robot_aim_pos = [0.5,0.1,0.3]       # 机器人初始位置
        self.sleeptime = 0
        self.aim_handler = []


    def isConnected(self):
        """
        判断是否已经与仿真环境连接
        """
        return sim.simxGetConnectionId(self.clientID) != -1


    def connect(self):
        """
        与仿真环境连接
        """
        sim.simxFinish(-1)         # 关闭所有打开的连接
        self.clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)      # 连接到V-REP
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)            # 开始仿真
        time.sleep(1)

        # 如果已经连接上仿真环境
        if self.clientID != -1:
            # 获取摄像头句柄
            res, self.camera_handler = sim.simxGetObjectHandle(self.clientID, 'kinect_rgb', sim.simx_opmode_oneshot_wait)
            if res != sim.simx_return_ok: 
                print('Vision_sensor:'+'Could not get handle to Camera')

            # 获取机械臂末端句柄
            res,self.arm_handler = sim.simxGetObjectHandle(self.clientID, 'ik_mov', sim.simx_opmode_oneshot_wait)

            # 获取机械臂末端当前位置
            self.robot_pos_now = sim.simxGetObjectPosition(self.clientID, self.arm_handler, -1, sim.simx_opmode_oneshot_wait)[1]

        np.random.seed(int(time.time()*1000)%10000)
        # r = np.asarray(range(6))
        # np.random.shuffle(r)

        r = np.asarray([0,1,2,3,4,5])
        for i in r.tolist():
            # 获得目标物体句柄
            res, aim_handler = sim.simxGetObjectHandle(self.clientID, 'k'+str(i+1), sim.simx_opmode_oneshot_wait)

            # 随机生成目标物体位姿
            xyz = (np.random.rand(3)-0.5)*0.1+[0.5,0,0]
            xyz[2] = 0.1
            a = np.random.rand(1)*5

            # 设置目标物体位姿
            if res == 0:
                sim.simxSetObjectQuaternion(self.clientID, aim_handler, -1, [a,a,a], sim.simx_opmode_oneshot_wait)
                sim.simxSetObjectPosition(self.clientID, aim_handler, -1, xyz, sim.simx_opmode_oneshot_wait)
                res,t = sim.simxGetObjectPosition(self.clientID, aim_handler, -1, sim.simx_opmode_streaming)
            res, _ = sim.simxGetObjectPosition(self.clientID, self.arm_handler, -1, sim.simx_opmode_streaming)
            self.aim_handler.append(aim_handler)
        # print(self.aim_handler)


    def getObjPos(self):
        """
        获取所有物体的位置
        """
        pos_all = np.zeros([6,3])
        for i in range(6):
            res, pos_all[i,:] = sim.simxGetObjectPosition(self.clientID, self.aim_handler[i], -1, sim.simx_opmode_buffer)
        return pos_all


    def moveObjPos(self,i):
        """
        将抓取到的物体移动到场外
        """
        sim.simxSetObjectPosition(self.clientID, self.aim_handler[i], -1, [-1,0,0.1], sim.simx_opmode_oneshot_wait)
        
    
    def disconnect(self):
        """
        关闭仿真环境，结束连接
        """
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        sim.simxFinish(self.clientID)


    def getImg(self):
        """
        获取图像
        """
        time.sleep(0.5)

        # print (self.h_camera)
        res, resolution, image = sim.simxGetVisionSensorImage(self.clientID, self.camera_handler, 0, sim.simx_opmode_oneshot_wait)
        if res == 0:
            original = np.array(image, dtype=np.uint8)
            original.resize([resolution[1], resolution[0], 3])
            # 图像翻转
            original = cv2.flip(original, 0)
            # 颜色格式转换
            # img_res = original
            img_res = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            img_pre = image
        else:
            print("无法获取图像")
            img_res = np.zeros([200,200])
            img_pre = np.zeros([200,200])
        return res, img_res, img_pre
        
    
    def xyzRange(self, xyz):
        for i in range(3):
            xyz[i] = max(xyz[i], self.range[i][0])
            xyz[i] = max(xyz[i], self.range[i][1])
        return xyz
    
    
    def posAimErr(self):
        """
        判断机械臂末端当前位置与目标位置的误差。完成返回1
        """
        res, self.robot_pos_now1 = sim.simxGetObjectPosition(self.clientID, self.arm_handler, -1, sim.simx_opmode_buffer)
        err = (self.robot_pos_aim[0] - self.robot_pos_now1[0])**2 +\
              (self.robot_pos_aim[1] - self.robot_pos_now1[1])**2 +\
              (self.robot_pos_aim[2] - self.robot_pos_now1[2])**2
        err_f = err < 10**-6
        sleep_f = self.sleeptime < 0
        return err_f and sleep_f


    def moveStep(self):
        """
        机械臂末端运动到目标位置。
        采用插值形式实现连续控制，每次运动move_step_speed，需要在函数外反复调用并检测是否到达指定位置
        """
        err_sum = 0
        for i in range(3):
            err_i = self.robot_pos_aim[i] - self.robot_pos_now[i]
            err_sum = err_sum + err_i**2
        err_sum = err_sum**0.5

        for i in range(3):
            if(err_sum > self.move_step_speed):
                self.robot_pos_now[i] = self.robot_pos_now[i] + self.move_step_speed*(self.robot_pos_aim[i]-self.robot_pos_now[i])/err_sum
            else:
                self.robot_pos_now[i] = self.robot_pos_aim[i]

        # 设置机械臂关节到目标角度
        sim.simxSetObjectPosition(self.clientID, self.arm_handler, -1, self.robot_pos_now, sim.simx_opmode_oneshot)
        self.sleeptime -= 0.05


    def handOpen(self, isOpen):
        """
        爪子张开
        """
        # 控制vrep中爪子开合
        ret = sim.simxSetFloatSignal(self.clientID, "gripper", isOpen, sim.simx_opmode_oneshot_wait)


    def getImgFeature(self):
        """
        获取图像中每个颜色物体的特征（x, y, w, h, a, per），6×6
        """
        res, img_res, img_pre = self.getImg()
        # cv2.imshow('original_out', original_out)
        
        if res == 0:
            image = img_res

            # 阈值化识别rgb色块
            # inRange函数将低于lower和高于upper的部分分别变成0，lower～upper之间的值变成255
            img_green  = cv2.inRange(img_res, np.array([224,0,0]), np.array([255,32,32]))      # 蓝色
            img_red    = cv2.inRange(img_res, np.array([0,224,0]), np.array([32,255,32]))      # 绿色
            img_blue   = cv2.inRange(img_res, np.array([0,0,224]), np.array([32,32,255]))      # 红色
            img_pink   = cv2.inRange(img_res, np.array([0,224,224]), np.array([32,255,255]))   # 黄色
            img_yellow = cv2.inRange(img_res, np.array([224,0,224]), np.array([255,32,255]))   # 粉色
            img_cyan   = cv2.inRange(img_res, np.array([224,224,0]), np.array([255,255,32]))   # 青色
            
            img_input = np.zeros([6,6])

            # 最小目标颜色的距离初始化
            i = 0
            aim_color_p = [999,999]
            find_flag = 0
            for color_i in [img_green, img_red, img_blue, img_pink, img_yellow, img_cyan]:
                # 轮廓检测
                contours, hierarchy = cv2.findContours(color_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                for contours_i in range(len(contours)):
                    # 轮廓面积滤波，去除小色块
                    area = cv2.contourArea(contours[contours_i])
                    # 通过测试，目标的面积约3200左右，取阈值2500，进行判断
                    if(area > 2500):
                        # 绘制所有大尺寸的轮廓，黑色，目标轮廓后面进行覆盖
                        # cv2.drawContours(images,contours[contours_i],-1,(0,0,0),3)
                        # rect[0]返回矩形的中心点
                        # rect[1]返回矩形的长和宽
                        # rect[2]返回矩形的旋转角度
                        rect = cv2.minAreaRect(contours[contours_i])
                        w = rect[1][0]
                        h = rect[1][1]
                        x = rect[0][0] - 256
                        y = rect[0][1] - 256
                        a = rect[2]
                        if w > h:
                            a = a+90
                        per = area/(w*h)       # 轮廓面积与方形面积之比
                        find_flag = 1
                        
                if find_flag == 1:
                    img_input[i] = [x, y, w, h, a, per]
                    boxp = cv2.boxPoints(rect).astype(np.int)
                    cv2.drawContours(image, [boxp], -1, (0,0,0), 3)
                    find_flag = 0

                i = i + 1
                
            cv2.imshow('Camera', image)
            cv2.waitKey(1)
            # print(np.asarray(img_input))
            return np.asarray(img_input)