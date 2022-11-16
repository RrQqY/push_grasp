'''
torch = 0.41
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
# import gym
import time
import argparse

# 导入机器人仿真环境控制
import robot as Robot
import env as Env


#####################  hyper parameters  ####################
LR_A = 0.001    # Actor学习率
LR_C = 0.002    # Critic学习率
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement

TAU = 0.01
RENDER = False
# RENDER = 1
# ENV_NAME = 'Pendulum-v1'


###############################  DDPG  ####################################
class ANet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(ANet,self).__init__()
        hidden_dim = 30        # 隐藏层维度（30）
        self.fc1 = nn.Linear(s_dim, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)       # initialization
        self.out = nn.Linear(hidden_dim, a_dim)
        self.out.weight.data.normal_(0, 0.1)       # initialization


    def forward(self,x):
        """
        前向传递函数
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions_value = x*1.2       # 乘的系数指定了最大移动范围
        return actions_value


class CNet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNet,self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1) # initialization
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1) # initialization
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization


    def forward(self,s,a):
        """
        前向传递函数
        """
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x + y)
        actions_value = self.out(net)
        return actions_value


# ddpg是一个示例模板，有标准输入输出
# 本项目使用ddpg输入36维数据，输出推动的两个端点
# 增加了网络的保存和读取功能
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.memory_size = 1000        # 记忆库的大小
        self.memory = np.zeros((self.memory_size, s_dim * 2 + a_dim + 1), dtype = np.float32)
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim,a_dim)
        self.Actor_target = ANet(s_dim,a_dim)
        self.Critic_eval = CNet(s_dim,a_dim)
        self.Critic_target = CNet(s_dim,a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr = LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr = LR_A)
        self.loss_td = nn.MSELoss()
        self.batch_size = 32


    def choose_action(self, s):
        """
        选择动作。s是观测值
        """
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s)[0].detach() # ae（s）


    def loadpkl(self,path="p_pkl"):
        """
        加载训练好的模型
        """
        self.Actor_eval = torch.load(path+'/DDPG_Actor_eval.pkl')
        self.Actor_target = torch.load(path+'/DDPG_Actor_target.pkl')
        self.Critic_eval = torch.load(path+'/DDPG_Critic_eval.pkl')
        self.Critic_target = torch.load(path+'/DDPG_Critic_target.pkl')
        

    def savepkl(self):
        """
        存储模型
        """
        torch.save(self.Actor_eval,'p_pkl0/DDPG_Actor_eval.pkl')
        torch.save(self.Actor_target,'p_pkl0/DDPG_Actor_target.pkl')
        
        torch.save(self.Critic_eval,'p_pkl0/DDPG_Critic_eval.pkl')
        torch.save(self.Critic_target,'p_pkl0/DDPG_Critic_target.pkl')
    

    def store_transition(self, s, a, r, s_):
        """
        将信息存储到经验池
        """
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


    def update(self):
        """
        更新网络
        """
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        #self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct

        indices = np.random.choice(self.memory_size, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs,a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q) 
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)        # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_,a_)    # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br + GAMMA*q_           # q_target = 负的
        #print(q_target)
        q_v = self.Critic_eval(bs,ba)
        #print(q_v)
        td_error = self.loss_td(q_target,q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()



# dqn是一个示例模板，有标准输入输出

# 本项目使用dqn输入36维数据，输出最优抓取目标的编号
# 增加了网络的保存和读取功能
class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 50)        # input
        self.fc1.weight.data.normal_(0, 0.01)     # 运用二次分布随机初始化，以得到更好的值
        self.out = nn.Linear(50, n_actions)       # output
        self.out.weight.data.normal_(0, 0.01)
 

    def forward(self, x):
        """
        前向传递函数
        """
        x = self.fc1(x)
        x = nn.functional.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN():
    def __init__(self,
                 n_states,
                 n_actions,
                 batch_size = 32,            # 小批量梯度下降
                 learning_rate = 0.05,
                 epsilon = 0.9,
                 gamma = 0.9,
                 target_replace_iter = 4,
                 memory_size = 2000):        # 经验池的大小

        # 生成两个结构相同的神经网络eval_net和target_net
        self.eval_net, self.target_net = Net(n_states, n_actions),\
                                         Net(n_states, n_actions)
        self.n_states = n_states                       # 状态维度（输入维度）
        self.n_actions = n_actions                     # 可选动作数（输出维度）
        self.batch_size = batch_size                   # 小批量梯度下降，每个“批”的size
        self.learning_rate = learning_rate             # 学习率
        self.epsilon = epsilon                         # e-greedy系数
        self.gamma = gamma                             # 回报衰减率
        self.memory_size = memory_size                 # 记忆库的规格
        self.taget_replace_iter = target_replace_iter  # target网络延迟更新的间隔步数
        self.learn_step_counter = 0                    # 在计算隔n步跟新的的时候用到，说明学习到多少步了
        self.memory_counter = 0                        # 用来计算存储索引
        self.memory = np.zeros((self.memory_size, self.n_states * 2 + 2))  # 初始化记忆库，存储两个state和reward和action
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)    # 网络优化器（Adam）
        self.loss_func = nn.MSELoss()                  # 网络的损失函数（MSE）


    def loadpkl(self,path="g_pkl"):
        """
        加载训练好的模型
        """
        self.eval_net=torch.load(path+'/dqneval_net.pkl')
        self.target_net=torch.load(path+'/dqntarget_net.pkl')


    def savepkl(self):
        """
        存储模型
        """
        torch.save(self.eval_net,'g_pkl0/dqneval_net.pkl')
        torch.save(self.target_net,'g_pkl0/dqntarget_net.pkl')


    def choose_action(self, x, in_obj_list):
        """
        选择动作。x是观测值
        """
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:            # 有epsilon概率由网络选择动作，选择actions_value最大的动作
            actions_value = self.eval_net.forward(x)
            action = int(torch.max(actions_value, 1)[1])
        else:                                             # （1-epsilon）概率随机选择动作
            action = int(np.random.randint(0, self.n_actions))

        while not (action in in_obj_list):        # 如果选择的动作是不在场地内的物块，则重新选择
            action = int(np.random.randint(0, self.n_actions))
            # print('!!!action is not in obj_list')

        return action
 

    def store_transition(self, s, a, r, s_):
        """
        将信息存储到经验池
        """
        transition = np.hstack((s, [a, r], s_))           # 将信息捆在一起
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size    # 如果memory_counter超过了memory_size的上限，就重新开始索引
        self.memory[index, :] = transition                # 将信息存到相对应的位置
        self.memory_counter += 1
 

    def update(self):
        """
        更新网络
        """
        # 判断target net什么时候更新
        if self.learn_step_counter % self.taget_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())     # 将eval_net中的state更新到target_net中
 
        # 采用mini-batch去更新(从记忆库中随机抽取一些记忆)
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])
 
        # 获得q_eval、q_target，计算loss
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)
 
        # 将loss反向传递回去，更新eval网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main(args):
    """
    主进程
    """
    mode = int(args.mode)        # 运行模式

    env = Env.robot_env()
    ddpg = DDPG(env.n_actions_p, env.n_states, 0)
    dqn = DQN(env.n_states, env.n_actions_g)

    t1 = time.time()

    MAX_EPISODES = 2000           # 总迭代次数
    MAX_EP_STEPS = 7             # 每次迭代尝试的步数
    episode_i = 0

    # 控制参数
    flag_p = 1        # 推动
    flag_g = 1        # 抓取

    # mode1 训练推动
    if mode == 1:
        MAX_EP_STEPS = 1
        flag_p = 1
        flag_g = 0
    # mode2 测试推动
    elif mode == 2:
        MAX_EP_STEPS = 1
        flag_p = 1
        flag_g = 0
        ddpg.loadpkl("p_pkl0")
    # mode3 训练抓取（结合推动）
    elif mode == 3:
        MAX_EP_STEPS = 7
        flag_p = 1
        flag_g = 1
        ddpg.loadpkl()
    # mode4 总体测试
    elif mode == 4:
        MAX_EP_STEPS = 7
        ddpg.loadpkl()
        dqn.loadpkl()
        flag_p = 1
        flag_g = 1

    # con = np.zeros([200,200])

    no_change_count = 0
    num_in_env_old = 0
    out_obj_list = []

    for i in range(MAX_EPISODES):
        env.reset()                 # 重置初始状态
        s = env.getImgInput()       # 获取当前场景状态特征向量s

        # 如果场地已空，则重置为初始状态
        ep_reward = 0.0             # 当前迭代的总reward
        episode_i += 1
        s_p = s.copy()
        s_g = s.copy()

        for episode_step_i in range(MAX_EP_STEPS):
            # cv2.imshow("con", con)
            if flag_g == 1:
                num_in_env = 0
                in_obj_list = []

                pos_all = env.robot.getObjPos()      # 获取所有物体位置
                for i in range(6):
                    if (pos_all[i,0] > env.robot.range[0][0] and pos_all[i,0] < env.robot.range[0][1]) and \
                    (pos_all[i,1] > env.robot.range[1][0] and pos_all[i,1] < env.robot.range[1][1]):
                        num_in_env = num_in_env + 1
                        in_obj_list.append(i)
                # 检查是否和上回合一样
                if num_in_env == num_in_env_old:
                    no_change_count = no_change_count + 1
                    print("no_change_count = ", no_change_count)
                else:
                    num_in_env_old = num_in_env
                    no_change_count = 0
                # 如果场地已空或连续多回合场上物块数没有变化，则重置
                if(num_in_env == 0) or (no_change_count >= 4):
                    env.reset()
                    s = env.getImgInput()       # 获取当前场景状态特征向量s
                    no_change_count = 0

                    pos_all = env.robot.getObjPos()      # 获取所有物体位置
                    for i in range(6):
                        if (pos_all[i,0] > env.robot.range[0][0] and pos_all[i,0] < env.robot.range[0][1]) and \
                        (pos_all[i,1] > env.robot.range[1][0] and pos_all[i,1] < env.robot.range[1][1]):
                            num_in_env = num_in_env + 1
                            in_obj_list.append(i)
                
                print("@@@num_in_env = ", num_in_env)
                print("@@@in_obj_list = ", in_obj_list)

            # 推动部分强化学习
            push_flag = env.getPushFlag(s_g)              # 判断是否需要推动

            if flag_p == 1 and push_flag == 1:
                s_p = env.getImgInput()                   # 获取当前场景状态特征向量s
                # cv2.imshow("con", con)
                a_p = ddpg.choose_action(s_p)             # 通过DDPG选择一个推动起点和目标点

                # a_p = np.clip(a_p, -1, 1)    
                s_, r, done, info = env.step_push(a_p)    # 执行一次推动动作，获取下一状态s_，reward，是否结束标记done

                ddpg.store_transition(s_p, a_p, r, s_)    # 将一步的信息存入经验池

                if ddpg.pointer > ddpg.memory_size:       # 当经验池存满（非必要等到存满）的时候，开始训练
                    if mode == 1:
                        ddpg.update()
                s_p = s_
                ep_reward += r

                print('总模拟:', episode_i, "推动", episode_step_i,' Reward: %.2f' % ep_reward, 'action:', a_p)

                time.sleep(0.5)

            # 抓取部分强化学习
            if flag_g == 1:
                s_g = env.getImgInput()       # 获取当前场景状态特征向量s
                a_g = dqn.choose_action(s_g, in_obj_list)   # 通过DQN选择抓取目标物体
                s_, r, done = env.step_grasp(a_g)           # 执行一次抓取动作，获得下一个状态s_，回报r，是否结束标记done

                print("抓取 r a" , r, a_g)
                print("\n模拟", episode_i,"抓取", episode_step_i, "动作", a_g)

                dqn.store_transition(s_g, np.asarray(a_g) , r, s_)     # 将一步的信息存入经验池

                if dqn.memory_counter > dqn.memory_size:               # 当经验池存满（非必要等到存满）的时候，开始训练
                    if mode == 3:
                        dqn.update()
                # if done:                    # 如果done（智能到达终点/掉入陷阱），结束本轮
                #     break
                s_g = s_

                time.sleep(0.5)

    # 保存模型
    if episode_step_i == MAX_EP_STEPS - 1:
        print("ddpg.savepkl()")
        if mode == 1:
            ddpg.savepkl()
        if mode == 3:
            dqn.savepkl()
        
    print('Running time: ', time.time() - t1)


# 程序入口
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Push-Grasping')

    # --------------- 参数设定 ---------------
    parser.add_argument('--mode', default = 4, type = int, help='run in which mode?')

    # 开始主程序
    args = parser.parse_args()
    main(args)