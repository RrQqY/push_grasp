import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
# import gym
import time

import robot as Robot

class robot_env():
    """
    仿真环境类
    """
    def __init__(self):
        self.state = 0                     # 当前环境状态
        self.robot = None
        self.n_states = 36                 # 状态维度（输入维度）
        self.n_actions_p = 4               # 可选推动动作维度（输出维度）
        self.n_actions_g = 6               # 可选抓取动作维度（输出维度）

        self.obj_num = 6                    # 目标个数
        self.dis_all = np.zeros([6,6])
        self.img_dis_all = np.zeros([6,6])


    def reset(self):
        """
        重置场景为初始状态
        """
        self.state = 1
        self.robot = Robot.Robot()
        self.robot.connect()
        # s = self.getImgInput()
        # return s


    def getImgInput(self):
        """
        将机械臂末端移动到初始位置并张开爪子，然后获取图像并获得当前场景中物块特征信息向量 6×6
        """
        # 如果当前状态为1（初始状态）
        if(self.state == 1 or 1):
            self.robot.robot_pos_aim = self.robot.robot_aim_pos.copy()
            
            while True:
                self.robot.moveStep()           # 移动到初始位置
                if self.robot.posAimErr():      # 如果到达目标点
                    self.state = 2              # 进入状态2（获取图像）
                    # 获取图像并获得当前场景中物块特征信息向量 6×6
                    self.imgInput = self.robot.getImgFeature()

                    self.robot.handOpen(0)      # 爪子打开
                    # time.sleep(1)
                    break

            return self.imgInput.reshape(36)


    def getDistReward(self, d_all1, d_all2):
        """
        计算推动导致物体间距变大的距离奖励。
        若推动之前距离就很大，则不计入奖励
        """
        d1 = d_all1
        d2 = d_all2

        far_threshold = 0.1

        r_d = np.zeros([1,6])
        r_d = np.abs(d1 - d2)

        for i in range(6):
            if d1[i] > far_threshold and d2[i] > d1[i]:
                r_d[i] = 0

        r = np.sum(r_d)
        # print("!!!r_d", r_d)
        return r


    def step_push(self, a):
        """
        执行一次推动动作。
        a为动作目标，a[0] a[1]为第一个点，a[2] a[3]为第二个点。
        返回下一个状态s_，回报r，是否结束标记done
        """
        self.pos_all = self.robot.getObjPos()

        for i in range(self.obj_num):
            # 所有物体的位置减去当前物体的位置，对相对位置求二范数，并升序排序
            # 得到每个物体与其他物体的二范数距离（列，升序排序）
            self.dis_all[:,i] = np.sort(np.linalg.norm(self.pos_all-self.pos_all[i,:], axis=1))
        # print("dis_all",self.dis_all)
        
        d1 = 2*np.sum(self.dis_all[1,:])         # 所有物体与最近物体的二范数距离
        d_all1 = self.dis_all[1,:].copy()
        h1 = np.average(self.pos_all[:,2])       # 所有物体处在的的平均高度

        # 动作顺序：state2获取图像，state3对齐目标物体，state10下落到第一个点，state11移动到第二个点，state99完成推动
        while True:
            done = 0 
            # 对齐抓取
            if(self.state == 2):        # 如果为状态2，获取完图像
                self.robot.handOpen(1)
                self.robot.sleeptime = 0.1
                self.state = 3          # 进入状态3，开始对齐目标物体

            if(self.state == 3):
                if self.robot.posAimErr():
                    self.state = 10     # 进入状态10，下落到第一个点

            if(self.state == 10):
                x, y = a[0], a[1]
                k = 0.1
                self.robot.robot_pos_aim[0] = 0.5 + x*k
                self.robot.robot_pos_aim[1] = 0 + y*k
                self.robot.robot_pos_aim[2] = self.robot.move_height[2]
                
                if self.robot.posAimErr():
                    self.state = 11     # 进入状态11，移动到第二个点

            if(self.state == 11):
                x, y = a[2], a[3]
                k = 0.1
                self.robot.robot_pos_aim[0] = 0.5 + x*k
                self.robot.robot_pos_aim[1] = 0 + y*k
                self.robot.robot_pos_aim[2] = self.robot.move_height[2]

                if self.robot.posAimErr():
                    self.state = 99     # 进入状态99，推动已完成
                    break

            self.robot.moveStep()                    # 移动一步
            self.pos_all = self.robot.getObjPos()    # 获取所有物体位置
            # print(pos_all)
            # print("state",self.state)
            # print("robot.robot_pos_aim",self.robot.robot_pos_aim)
            time.sleep(0.05)

        time.sleep(0.1)

        self.pos_all = self.robot.getObjPos()
        # r=np.asarray(np.sum(-(self.pos_all[:,0]-0.4)))
        # r=np.sum(self.pos_all[:,0]<0)
        
        ### 计算回报r
        for i in range(self.obj_num):
            self.dis_all[:,i] = np.sort(np.linalg.norm(self.pos_all-self.pos_all[i,:], axis=1))
            # dis_all[:,i]=np.exp(dist-0.2)-1

        d2 = 2*np.sum(self.dis_all[1,:])        # 推动动作后，所有物体与最近物体的二范数距离
        d_all2 = self.dis_all[1,:].copy()
        h2 = np.average(self.pos_all[:,2])      # 所有物体处在的的平均高度

        r_dist = self.getDistReward(d_all1, d_all2)     # 距离奖励

        if abs(d2 - d1) < 0.01:             # 如果推动动作后，所有物体与最近物体的二范数距离没有变化
            r = - 100
        else:
            r = 10*r_dist                # 推动导致物体间距变大，带来的收益r
            print("####r_dist",10*r_dist)

        if not abs(h1 - h2) < 0.1:
            r = r + 200*(h1 - h2)           # 推动导致物体变不重叠，带来的收益r

        # 移动距离惩罚
        move_dist = ((float(a[2]) - float(a[0]))**2 + (float(a[3]) - float(a[1]))**2)**0.5
        move_dist_threshold = 3.2
        if move_dist > move_dist_threshold:
            if abs(d2 - d1) < 0.01:
                r = r - (move_dist-move_dist_threshold)*4     # 推动距离太远，带来的惩罚r
                print("TTTTT, too far")
            else:
                r = r - (move_dist-move_dist_threshold-0.2)*4     # 推动距离太远，带来的收益r
                print("TTTTT, too far, but good")
        

        # print("DDD", 10*(d2 - d1), "HHH", 200*(h1 - h2))

        # 如果为状态99，推动已完成
        if self.state == 99:
            # self.robot.disconnect()
            done = 1
        else:
            done = 0

        # 获取当前状态s_
        s_ = self.getImgInput()

        return s_, r, done, 0


    def getPushFlag(self, s):
        """
        获得推动判断分数，判断是否需要推动
        """
        sout_t = s.reshape([6, 6])
        sout_txy = sout_t[:, :2]
        leni = 2*np.abs(sout_t[:, 0])
        leni_sum = np.sum(leni > 0.1)
        # print("leni_sum",leni_sum,sout_txy)
        for i in range(leni_sum):
            self.img_dis_all[:,i] = np.sort(np.linalg.norm(sout_txy-sout_txy[i,:], axis=1))

        if leni_sum > 1:
            r = np.min(self.img_dis_all[1, :leni_sum-1])
        else:
            r = 99

        print("r", r, self.img_dis_all[1,:])

        if r < 60:
            push_flag = 1
        else:
            push_flag = 0

        return push_flag
    
    
    def step_grasp(self, a):
        """
        执行一次抓取动作。
        a为选择的动作编号（抓取哪个物体）。
        返回下一个状态s_，回报r，是否结束标记done
        """
        # 动作顺序：state2获取图像，state3对齐目标物体，state9爪子下降并关闭，state12爪子抬起，state99完成推动
        while True:
            done = 0
            # 对齐抓取
            if(self.state == 2):        # 如果状态2，获取完图像
                if len(self.imgInput) > 0:
                    aim_color_p = self.imgInput[a]         # 要抓取的物块的特征

                    if(abs(aim_color_p[0]) < 999):
                        # print(aim_color_p)

                        # 移动至目标物块上方
                        k = 0.5/512
                        self.robot.robot_pos_aim[0] = 0.5 + (aim_color_p[1])*k
                        self.robot.robot_pos_aim[1] = 0 + (aim_color_p[0])*k
                        # k2=0.3/10
                        # maxindex = np.asarray(a)
                        # self.robot.robot_pos_aim[0] =0.5+(maxindex%10-5)*k2
                        # self.robot.robot_pos_aim[1] =0+(maxindex//10-5)*k2
                        self.state = 3    # 进入状态3，对齐目标物体
                else:
                    self.state = 99

            if(self.state == 3):
                # print(self.state)
                if self.robot.posAimErr():
                    self.state = 9        # 进入状态9，爪子已对齐目标物体，对齐后爪子下降，到达最低点后爪子关闭
            
            if(self.state == 9):
                self.robot.robot_pos_aim[2] = self.robot.move_height[0]    # 爪子下降
               
                if self.robot.posAimErr():
                    self.robot.handOpen(1)     # 爪子关闭
                    self.robot.sleeptime = 1
                    self.state = 12            # 进入状态12，爪子抬起

            if(self.state == 12 and self.robot.posAimErr()):
                self.robot.robot_pos_aim[2] = self.robot.move_height[1]

                if self.robot.posAimErr():       # 如果爪子已经抬起
                    self.pos_all = self.robot.getObjPos()    # 获取所有物体位置

                    for i in range(6):
                        if self.pos_all[i,2] > 0.2:
                            self.robot.moveObjPos(i)         # 检查z轴高于0.2的物体，将其移动至场地外
                        
                    self.robot.handOpen(0)     # 爪子打开
                    self.robot.sleeptime = 2

                    self.state = 99
                    break
            
            self.robot.moveStep()
            self.pos_all = self.robot.getObjPos()    # 获取所有物体位置
            time.sleep(0.05)
        
        self.pos_all = self.robot.getObjPos()

        # 计算回报r
        r = np.asarray(np.sum(-(self.pos_all[:,0]-0.4)))

        if self.state == 99:    # 如果为状态99，抓取已完成
            done = 1
        else:
            done = 0

        # 获取当前状态s_
        s_ = self.getImgInput()

        return s_, r, done 