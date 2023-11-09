
import sys
sys.path.append('D:\\1_GitProject\\2_ML')

import random

import time
import numpy as np
import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm
from tqdm import tqdm
import gymnasium as gym
import os
import torch
import torch.nn.functional as F

import collections
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time


#from .. import rl_utils as rlu
import rl_utils as rlu
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("1_PPO_GAIL/log/") 

deb = True
is_bc = False
class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__() #BUG:++++++++++++++++++++++++漏加了
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))#BUG:F.->torch
        return F.softmax(self.fc2(x),dim=-1) #BUG:====不存在维度1，只有-1 ###!!!!!!!!!!!!!!!!
        #BUG:??????????????????????torch.softmax??

class ValueNet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)#BUG:???

def GAE_advantage(td_delta,gamma,lmbda):
    #BUG：根本问题：在循环内，你需要更新每个时间步的优势，advantage需要被索引，而不是累加到一个总优势上
    #--------------------------------------------------------------------------
    #advantage = torch.zeros() #BUG:shape>>>没有提供正确形状
    #for i in range(reversed(td_delta)):#BUG>>>>>reversed已经是反向迭代器了
    #    advantage += gamma * lmbda * advantage + i
    #return advantage #BUG:????????????????????????????
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #z-得分标准化
    td_delta = td_delta.detach().numpy()
    advantage_list=[]
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage += gamma*lmbda*advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = torch.tensor(np.array(advantage_list),dtype=torch.float)
    advantage_list = (advantage_list - advantage_list.mean()) / (advantage_list.std()+1e-5)
    return advantage_list
#agent = PPO(state_dim,hidden_dim,action_dim,gamma,lmbda,epochs,actor_lr,critic_lr,eps,device)    
class PPO:
    def __init__(self,state_dim,hidden_dim,action_dim,gamma,lmbda,epochs,actor_lr,critic_lr,eps,device): #BUG:lr1?lr2?
        self.actor = PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        #self.critic = ValueNet(state_dim,hidden_dim,action_dim)#BUG:action_dim = 1
        self.critic = ValueNet(state_dim,hidden_dim).to(device)#BUG:>>>>>+++++.to(device)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(),lr=actor_lr)#BUG:?torch.optim.Adam
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(),lr=critic_lr) #BUG:?Adam
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps #BUG:???eps
        self.device = device
    
    def take_action(self,state): 
        state = torch.tensor(np.array(state),dtype=torch.float).to(self.device) 
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs) 
        action = action_dist.sample().item()
        return action
    
    def update(self,transition_dict):
        #BUG:states = torch.tensor(transition_dict["states"],dtype=torch.float).to(self.device)
        #BUG:++++++提示应该先转换为单一numpy，再转换为torch张量
        states_array = np.array(transition_dict["states"])
        states = torch.tensor(states_array, dtype=torch.float).to(self.device)
        
        #BUG:actions = torch.tensor(transition_dict["actions"],dtype=torch.int).to(self.device) #BUG:++++++.view(-1,1)
        actions = torch.tensor(transition_dict["actions"],dtype=torch.int64).view(-1,1).to(self.device) #BUG:++++++.view(-1,1)
        rewards = torch.tensor(transition_dict["rewards"],dtype=torch.float).view(-1,1).to(self.device)
        
        next_states_array = np.array(transition_dict["next_states"])
        next_states = torch.tensor(next_states_array,dtype=torch.float).to(self.device)
        #BUG:budones = torch.tensor(transition_dict["dones"],dtype=torch.int).to(self.device) #BUG:?
        #BUG:truncated = torch.tensor(transition_dict["truncated"],dtype=torch.int).to(self.device)
        #BUG:.view(-1,1)问题++++++++++++++++++++由于神经网络返回一个[batch_size,1]的张量
        dones = torch.tensor(transition_dict["dones"],dtype=torch.int).view(-1,1).to(self.device) #BUG:?
        truncated = torch.tensor(transition_dict["truncated"],dtype=torch.int).view(-1,1).to(self.device)
        
        td_target = rewards + self.gamma * self.critic(next_states)*(1-dones|truncated) #BUG:++++++++遗漏(1-done|truncated)
        td_delta = td_target - self.critic(states)

        #BUG:advantage = GAE_advantage(td_delta,self.gamma,self.lmbda,self.device)
        #BUG:++++++遗漏.to(device)转移设备 +++++++++++++++错误：td_delta还在cuda里，要转移到cpu中！！
        #ERROR:advantage = GAE_advantage(td_delta.cpu(),self.gamma,self.lmbda).to(self.device)
        advantage = rlu.compute_advantage(self.gamma,self.lmbda,td_delta.cpu()).to(self.device)
        
        #BUG:old_log_probs = torch.log(self.actor(states)).gather(1,actions).to(self.device) #BUG:?
        #--------------++++++++++++++彻头彻尾大错误！！！log应该在外层，减少计算量
        old_log_probs = torch.log(self.actor(states).gather(1,actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1,actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio,1-self.eps,1+self.eps) * advantage
            
            #BUG：没有归一化，方向还搞错了++++++++++++++++++++++++++++++++++++
            #actor_loss = torch.min(surr1,surr2)
            #critic_loss = None#BUG:>>>>>F.mse_loss 均方差损失，需要detach目标
            actor_loss = torch.mean(-torch.min(surr1,surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states),td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

actor_lr = 1e-3
critic_lr = 1e-2
lmbda = 0.95
gamma = 0.98
total_epochs = 10
total_episodes = 500

eps = 0.2
epochs = 10
device = ("cuda" if torch.cuda.is_available() else "cpu")

env_name = "CartPole-v1"
env = gym.make(env_name)
#BUG:seed???????+++++++++++++++
torch.manual_seed(0)

state_dim = env.observation_space.shape[0] #BUG:????+++++++观测空间
hidden_dim = 128
action_dim = env.action_space.n #BUG:????+++++++动作空间

agent = PPO(state_dim,hidden_dim,action_dim,gamma,lmbda,epochs,actor_lr,critic_lr,eps,device)

CKP_PATH = '1_PPO_GAIL/tmp/E_v2.pt'
BEST_CKP_PATH = '1_PPO_GAIL/output/E_v2.pt'

def read_ckp(agent,CKP_PATH):
    if os.path.exists(CKP_PATH):
        tmp = torch.load(CKP_PATH)
        agent.gamma = tmp.get("gamma",None)
        agent.lmbda = tmp.get("lmbda",None)
        agent.eps = tmp.get("eps",None)
        agent.actor.load_state_dict(tmp.get("actor_weight",None))
        agent.critic.load_state_dict(tmp.get("critic_weight",None))
        return tmp.get("epoch",0),tmp.get("episode",0),tmp.get("return_list",[])
    else:
        os.makedirs(os.path.dirname(CKP_PATH), exist_ok=True)
        return 0,0,[]

def save_ckp(data, PATH):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)  # 确保路径存在
    torch.save(data, PATH)

#s_epoch,s_episode,return_list = read_ckp(agent,BEST_CKP_PATH)
s_epoch,s_episode,return_list = read_ckp(agent,CKP_PATH)
rlu.picture_return(return_list,"PPO","CarPort-v1",9)

#s_epoch = 0
#s_episode = 0
#return_list = []
def train_on_policy(writer,env,agent,s_epoch,total_epochs,s_episode,total_episodes,return_list,CKP_PATH,BEST_CKP_PATH):
    start_time = time.time()
    best_score = -1e10
    if return_list is None:
        return_list = []
    for epoch in range(s_epoch,total_epochs):
        with tqdm(total=(total_episodes - s_episode),desc='<%d/%d>'%(epoch+1,total_epochs),leave=False) as pbar:
            for episode in range(s_episode,total_episodes):
                episode_return = 0
                transition_dict={
                    "states":[],"rewards":[],"next_states":[],"actions":[],"dones":[],"truncated":[]
                }
                done = truncated = False
                state = env.reset()[0]#BUG:>>>>??
                while not (done | truncated):
                    action = agent.take_action(state)
                    next_state,reward,done,truncated,info = env.step(action)
                    transition_dict["states"].append(state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["actions"].append(action)
                    transition_dict["dones"].append(done)
                    transition_dict["truncated"].append(truncated)
                    state = next_state
                    episode_return +=reward

                return_list.append(episode_return)
                agent.update(transition_dict)

                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (total_episodes * epoch + episode + 1),
                                      'recent_return': '%.3f' % np.mean(return_list[-10:])})

                if episode_return >= best_score:
                    actor_best_weight = agent.actor.state_dict() #BUG:???
                    critic_best_weight = agent.critic.state_dict()
                    best_score = episode_return
                    best_point = {
                        'epoch': epoch,
                        'episode':episode,
                        'gamma':agent.gamma,
                        'lmbda':agent.lmbda,
                        'eps':agent.eps,
                        'actor_weight':actor_best_weight,
                        'critic_weight':critic_best_weight,
                        'return_list':return_list
                    }
                    save_ckp(best_point,BEST_CKP_PATH)
                


                # 添加到 TensorBoard
                writer.add_scalar('Episode Return', episode_return, episode + epoch * total_episodes)
                writer.add_scalar('PPO Return', episode_return, len(return_list))

                check_point = {
                     'epoch': epoch,
                     'episode':episode,
                     'gamma':agent.gamma,
                     'lmbda':agent.lmbda,
                     'eps':agent.eps,
                     'actor_weight':agent.actor.state_dict(),
                     'critic_weight':agent.critic.state_dict(),
                     'return_list':return_list
                 }
                save_ckp(check_point,CKP_PATH)
                pbar.update(1)
            s_episode = 0
    try:
        agent.actor.load_state_dict(actor_best_weight)
        agent.critic.load_state_dict(critic_best_weight)
        best_point = {
            'epoch': epoch,
            'episode':episode,
            'gamma':agent.gamma,
            'lmbda':agent.lmbda,
            'eps':agent.eps,
            'actor_weight':actor_best_weight,
            'critic_weight':critic_best_weight,
            'return_list':return_list
        }
        save_ckp(best_point,BEST_CKP_PATH)
    except:
        raise "Error:load best weights failed!"
    end_time = time.time()
    print('总耗时: %d分钟' % ((end_time - start_time)/60))

    return return_list

#train_on_policy(writer,env,agent,s_epoch,total_epochs,s_episode,total_episodes,return_list,CKP_PATH,BEST_CKP_PATH)

#writer.close()  # 关闭 TensorBoard writer

#return_list = []
#rlu.train_on_policy_agent(env,agent,0,10,0,50,return_list,CKP_PATH)
#rlu.picture_return(return_list,"PPO","CartPole",9)


#rlu.show_gym_policy(env_name, agent, render_mode='human', epochs=10, steps=500, model_type='AC', if_return=False)

def sample_expert_data(env,agent,n_episode):
    states = []
    actions = []
    for _ in range(n_episode):
        state = env.reset()[0]
        done = truncated = False
        while not (done | truncated):
            action = agent.take_action(state)
            next_state,_,done,truncated,_ = env.step(action)
            states.append(state)
            actions.append(action)
            state = next_state
    return np.array(states),np.array(actions)

if is_bc:
    n_episode = 4
    expert_s,expert_a = sample_expert_data(env,agent,n_episode)
    #BUG:random index???
    n_samples = 1000
    #BUG:ERROR___random_index = np.random.sample(range(expert_s.shape[0]),n_samples)
    random_index = np.random.choice(range(expert_s.shape[0]), n_samples, replace=False)
    expert_s,expert_a = expert_s[random_index],expert_a[random_index]

class BehaviorClone:
    def __init__(self,state_dim,hidden_dim,action_dim,policy_lr,device):
        self.policy = PolicyNet(state_dim,hidden_dim,action_dim).to(device) 
        self.optimizer = torch.optim.Adam(params=self.policy.parameters(),lr=policy_lr)
        self.device = device

    def take_action(self,state):
        state = torch.tensor(np.array(state),dtype=torch.float).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action
    
    def learn(self,states,actions):
        states = torch.tensor(np.array(states),dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(actions),dtype=torch.int64).view(-1,1).to(self.device)
        log_porbs = torch.log(self.policy(states).gather(1,actions))
        bc_loss = torch.mean(-log_porbs)

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

if is_bc:
    bc_lr = 1e-3
    bc_agent = BehaviorClone(state_dim,hidden_dim,action_dim,bc_lr,device)

def test_agent(env,agent,n_episode = 5):
    return_list = []
    for _ in tqdm(range(n_episode),leave=False):
        episode_return = 0.0
        state = env.reset()[0]
        done = truncated = False
        while not (done | truncated):
            action = agent.take_action(state)
            next_state,reward,done,truncated,_ = env.step(action)
            episode_return += reward
            state = next_state
        return_list.append(episode_return)
    return np.mean(return_list)



def train_bc_policy(env,agent,expert_s,expert_a,n_iterations,batch_size,n_episode = 5,return_list = None,BC_CKP_PATH=None):
    if return_list == None:
        return_list = []
    with tqdm(total=n_iterations,leave=False) as pbar:
        for i in range(n_iterations):
            # 更新进度条描述
            pbar.set_description("<%d/%d>" % (i+1, n_iterations))

            #==================
            sample_index = np.random.randint(low = 0,high=expert_s.shape[0],size=batch_size)
            agent.learn(expert_s[sample_index],expert_a[sample_index])
            current_return = test_agent(env,agent,10)
            return_list.append(current_return)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

if is_bc:
    n_iterations = 100
    batch_size = 100
    bc_return_list = []
    n_episode=5
    train_bc_policy(env,bc_agent,expert_s,expert_a,n_iterations,batch_size,n_episode,bc_return_list)
    #rlu.picture_return(bc_return_list,"BC","CartPole-v1",9)
    bc_return_list = rlu.show_gym_policy(env_name, bc_agent, render_mode='human', epochs=10, steps=500, model_type='AC', if_return=True)

class Discriminator(torch.nn.Module):
    def __init__(self,state_dim,hidden_state,action_state):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,1)
    
    def forward(self,s,a):
        cat = torch.cat([s,a],dim=1)
        x = F.relu(self.fc1(cat))
        return F.sigmoid(self.fc2(x))

class GAIL:
    def __init__(self,state_dim,hidden_state,action_state,agent,lr,device,*,action_space = action_dim):
        self.discriminator = Discriminator(state_dim,hidden_state,action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),lr=lr)
        self.agent = agent
        self.action_space = action_space
        self.device = device
    
    def learn(self,expert_s,expert_a,agent_transition_dict):
        expert_s = torch.tensor(np.array(expert_s),dtype = torch.float).to(self.device)
        expert_a = torch.tensor(np.array(expert_a),dtype = torch.int64).to(self.device) #BUG:是否需要.view(-1,1)

        agent_s = torch.tensor(np.array(agent_transition_dict["states"]),dtype=torch.float).to(self.device)
        agent_a = torch.tensor(np.array(agent_transition_dict["actions"]),dtype= torch.int64).to(self.device)
        #rewards = torch.tensor(np.array(agent_transition_dict["rewards"]),dtype=torch.float).to(self.device)

        #BUG:重大遗漏：独热编码+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        expert_a = F.one_hot(expert_a,num_classes=self.action_space).float()
        agent_a = F.one_hot(agent_a,num_classes=self.action_space).float()

        agent_prob = self.discriminator(agent_s,agent_a)
        expert_prob = self.discriminator(expert_s,expert_a)

        discriminator_loss = torch.nn.BCELoss()(agent_prob,torch.ones_like(agent_prob))\
                            +torch.nn.BCELoss()(expert_prob,torch.zeros_like(expert_prob))
        
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        #BUG:遗漏detach()方法++++++++++++++++++++++
        #agent_rewards = -torch.log(agent_prob).cpu().numpy()
        agent_rewards = -torch.log(agent_prob).detach().cpu().numpy()
        agent_transition_dict["rewards"] = agent_rewards

        self.agent.update(agent_transition_dict)

def train_GAIL_policy(env,expert,gail,s_epoch,total_epochs,s_episode,total_episodes,return_list,CKP_PATH,BEST_CKP_PATH):
    agent = gail.agent
    start_time = time.time()
    best_score = -1e10
    if return_list is None:
        return_list = []
    for epoch in range(s_epoch,total_epochs):
        with tqdm(total=(total_episodes - s_episode),desc='<%d/%d>'%(epoch+1,total_epochs),leave=False) as pbar:
            for episode in range(s_episode,total_episodes):
                episode_return = 0
                transition_dict={
                    "states":[],"rewards":[],"next_states":[],"actions":[],"dones":[],"truncated":[]
                }
                expert_s = []
                expert_a = []

                done = truncated = False
                state = env.reset()[0]#BUG:>>>>??
                while not (done | truncated):
                    action = agent.take_action(state)
                    next_state,reward,done,truncated,info = env.step(action)
                    transition_dict["states"].append(state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["actions"].append(action)
                    transition_dict["dones"].append(done)
                    transition_dict["truncated"].append(truncated)
                    expert_s.append(state)
                    expert_a.append(expert.take_action(state))
                    state = next_state
                    episode_return +=reward

                return_list.append(episode_return)
                # -----------------------------
                # agent.update(transition_dict)
                # +++++++++++++++++++++++++++++++++++++++++++
                gail.learn(expert_s,expert_a,transition_dict)

                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (total_episodes * epoch + episode + 1),
                                      'recent_return': '%.3f' % np.mean(return_list[-10:])})

                if episode_return > best_score:
                    actor_best_weight = agent.actor.state_dict() #BUG:???
                    critic_best_weight = agent.critic.state_dict()
                    best_score = episode_return
                    best_point = {
                        'epoch': epoch,
                        'episode':episode,
                        'gamma':agent.gamma,
                        'lmbda':agent.lmbda,
                        'eps':agent.eps,
                        'actor_weight':actor_best_weight,
                        'critic_weight':critic_best_weight,
                        'return_list':return_list
                    }
                    save_ckp(best_point,BEST_CKP_PATH)
                


                # 添加到 TensorBoard
                writer.add_scalar('Episode Return', episode_return, episode + epoch * total_episodes)
                writer.add_scalar('PPO Return', episode_return, len(return_list))

                check_point = {
                     'epoch': epoch,
                     'episode':episode,
                     'gamma':agent.gamma,
                     'lmbda':agent.lmbda,
                     'eps':agent.eps,
                     'actor_weight':agent.actor.state_dict(),
                     'critic_weight':agent.critic.state_dict(),
                     'return_list':return_list
                 }
                save_ckp(check_point,CKP_PATH)
                pbar.update(1)
            s_episode = 0
    try:
        agent.actor.load_state_dict(actor_best_weight)
        agent.critic.load_state_dict(critic_best_weight)
        best_point = {
            'epoch': epoch,
            'episode':episode,
            'gamma':agent.gamma,
            'lmbda':agent.lmbda,
            'eps':agent.eps,
            'actor_weight':actor_best_weight,
            'critic_weight':critic_best_weight,
            'return_list':return_list
        }
        save_ckp(best_point,BEST_CKP_PATH)
    except:
        raise "Error:load best weights failed!"
    end_time = time.time()
    print('总耗时: %d分钟' % ((end_time - start_time)/60))

    return return_list

#共用一套参数
expert = agent #PPO
new_agent = PPO(state_dim,hidden_dim,action_dim,gamma,lmbda,epochs,actor_lr,critic_lr,eps,device)
gail_lr = 1e-3
gail = GAIL(state_dim,hidden_dim,action_dim,new_agent,gail_lr,device)


new_CKP_PATH = '1_PPO_GAIL/tmp/GAIL_new_v3.pt'
new_BEST_CKP_PATH = '1_PPO_GAIL/output/GAIL_new_v3.pt'

n_s_epoch,n_s_episode,n_return_list = read_ckp(new_agent,new_CKP_PATH)
n_total_epochs = 5
n_total_episodes = 200
#train_GAIL_policy(env,expert,gail,n_s_epoch,n_total_epochs,n_s_episode,n_total_episodes,\
#                  n_return_list,new_CKP_PATH,new_BEST_CKP_PATH)

rlu.picture_return(n_return_list,"GAIL","CarPort-v1",9)



