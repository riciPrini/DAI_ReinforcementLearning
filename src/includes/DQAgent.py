import traci
import random
import statistics as stat

import numpy as np
import torch as T
from includes.deep_q_network import DeepQNetwork
from includes.replay_memory import ReplayBuffer

class DQAgent(object):
    """
        This is single agent class.
    """
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7   ,
                 lambda_penalty=0.1, alpha_throughput=0.1,replace=1000, 
                 algo=None, env_name=None, chkpt_dir='tmp/dqn', TLC_name="gneJ26"):
        self.gamma = gamma
        self.lambda_penalty = lambda_penalty
        self.alpha_throughput = alpha_throughput  
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)


        ## Defining cardinal parameters
        # Accumulated traffic queues in each direction
        self.no_veh_N = 0
        self.no_veh_E = 0
        self.no_veh_W = 0
        self.no_veh_S = 0

        # Accumulated wait time in each direction
        self.wait_time_N = 0
        self.wait_time_E = 0
        self.wait_time_W = 0
        self.wait_time_S = 0
        
        self.TLC_name = TLC_name
        self.threshold = 0

        self.acc_wait_time = []

        self.reward = 0

        self.switch_time = 0
        self.next_switch = 0


    def get_reward(self):

        current_wait_time = self.get_avg_wait_time()
        queue_length_penalty = self.get_queue_length() 

        throughput = self.get_throughput()

        if len(self.acc_wait_time) == 0:
            self.reward = 0
        else:
            self.reward = (
            (self.acc_wait_time[-1] - current_wait_time)  # Reducing waiting time
            - self.lambda_penalty * queue_length_penalty  # Queue penality
            + self.alpha_throughput * throughput          # throughput_alpha
        )
        
        self.acc_wait_time.append(current_wait_time)

        if len(self.acc_wait_time) > 3: # to reduce the array size  / keep only last three elements
            self.acc_wait_time = self.acc_wait_time[-3:]

        return self.reward
    
    def vehicle_counter(self,vehicle_wait_time,road_id,lanes):
        """
        Function that counts the number of waiting vehicles in the traffic light
        """

        if road_id == lanes[0]: #North
            self.no_veh_N += 1 
            self.wait_time_N += vehicle_wait_time                    
        elif road_id == lanes[1]: #East
            self.no_veh_E += 1 
            self.wait_time_E += vehicle_wait_time
        elif road_id == lanes[2]: #West
            self.no_veh_W += 1 
            self.wait_time_W += vehicle_wait_time
        elif road_id == lanes[3]: #South
            self.no_veh_S += 1 
            self.wait_time_S += vehicle_wait_time
        
    
        
    def get_avg_wait_time(self):
        """
        Function that return the avg wait time of vehicles in the traffic light. 
        Regardless the 2x2 or 3x3 envirorment this still work due to the same number of lane IDs.

        
        """
        phase_central = traci.trafficlight.getRedYellowGreenState(self.TLC_name)
            ####
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)
            vehicle_wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            # print("TLC_name: ",self.TLC_name,"Vehicle ID: ", vehicle_id, "Speed: ", vehicle_speed, "Road Id: ", road_id)

            #Count vehicle at the TLC junction
            if vehicle_speed < 1: # Count only stoped vehicles
                if self.TLC_name == "gneJ24":
                    self.vehicle_counter(vehicle_wait_time,road_id,["","eL_N","eS_P","E55"]) #North,East,West,South
                elif self.TLC_name == "gneJ25":
                    self.vehicle_counter(vehicle_wait_time,road_id,["","eT_U","eN_L","eK_O"]) #North,East,West,South
                elif self.TLC_name == "gneJ26":
                    self.vehicle_counter(vehicle_wait_time,road_id,["eB_I","eD_I","eH_I","eF_I"]) #North,East,West,South
                elif self.TLC_name == "gneJ27":
                    self.vehicle_counter(vehicle_wait_time,road_id,["eA_H","eI_H","-E9","eG_H"]) #North,East,West,South
                elif self.TLC_name == "gneJ28":
                    self.vehicle_counter(vehicle_wait_time,road_id,["eC_D","-E16","eI_D","eE_D"]) #North,East,West,South
                elif self.TLC_name == "gneJ29":
                    self.vehicle_counter(vehicle_wait_time,road_id,["E4","eC_B","eA_B","eI_B"]) #North,East,West,South
                elif self.TLC_name == "gneJ30":
                    self.vehicle_counter(vehicle_wait_time,road_id,["eI_F","eE_F","eG_F",""]) #North,East,West,South
                elif self.TLC_name == "gneJ31":
                    self.vehicle_counter(vehicle_wait_time,road_id,["E15","-E0","eF_E",""]) #North,East,West,South
        
        return stat.mean([self.wait_time_N, self.wait_time_E, self.wait_time_W, self.wait_time_S])

    def step(self, action, step):
        # print(type(action))
        # 1st APPLY the choosed action
        if action == 0:
            traci.trafficlight.setProgram(self.TLC_name, "wave_N")
            # traci.trafficlight.setRedYellowGreenState(self.TLC_name, "GGGgrrrrGGGgrrrr")
        elif action == 1:
            traci.trafficlight.setProgram(self.TLC_name, "wave_E")
        elif action == 2:
            # print("sono qui")
            traci.trafficlight.setProgram(self.TLC_name, "0")
            # traci.trafficlight.setRedYellowGreenState(self.TLC_name, "rrrrGGGgrrrrGGGg")


        # 2nd, find the Reward 
        #func will be implemented later
        reward = self.get_reward()
        
        # 3rd: find new state
        # return state, reward, info_dict
        state = self.get_state()
        
        #state = [self.wait_time_N, self.wait_time_E, self.wait_time_W, self.wait_time_S,
         #        self.no_veh_N, self.no_veh_E, self.no_veh_W, self.no_veh_S]
        
        # 4th, an info dict
        info = {
        "TLC_name": self.TLC_name,
        "wait_time_N": self.wait_time_N,
        "wait_time_E": self.wait_time_E,
        "wait_time_W": self.wait_time_W,
        "wait_time_S": self.wait_time_S,
        "avg_wait_time": self.get_avg_wait_time(),
        "queue_length": self.get_queue_length(),
        "throughput": self.get_throughput()
        }

        return state, reward, info


        #zero all the class variable

    def get_state(self):
        phase_central = traci.trafficlight.getRedYellowGreenState(self.TLC_name)
        ####
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)
            vehicle_wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            #print("Vehicle ID: ", vehicle_id, "Speed: ", vehicle_speed, "Road Id: ", road_id)

            #Count vehicle at the TLC junction
            if vehicle_speed < 1: # Count only stoped vehicles
                if self.TLC_name == "gneJ24":
                    self.vehicle_counter(vehicle_wait_time,road_id,["","eL_N","eS_P","E55"]) #North,East,West,South
                elif self.TLC_name == "gneJ25":
                    self.vehicle_counter(vehicle_wait_time,road_id,["","eT_U","eN_L","eK_O"]) #North,East,West,South
                elif self.TLC_name == "gneJ26":
                    self.vehicle_counter(vehicle_wait_time,road_id,["eB_I","eD_I","eH_I","eF_I"]) #North,East,West,South
                elif self.TLC_name == "gneJ27":
                    self.vehicle_counter(vehicle_wait_time,road_id,["eA_H","eI_H","-E9","eG_H"]) #North,East,West,South
                elif self.TLC_name == "gneJ28":
                    self.vehicle_counter(vehicle_wait_time,road_id,["eC_D","-E16","eI_D","eE_D"]) #North,East,West,South
                elif self.TLC_name == "gneJ29":
                    self.vehicle_counter(vehicle_wait_time,road_id,["E4","eC_B","eA_B","eI_B"]) #North,East,West,South
                elif self.TLC_name == "gneJ30":
                    self.vehicle_counter(vehicle_wait_time,road_id,["eI_F","eE_F","eG_F",""]) #North,East,West,South
                elif self.TLC_name == "gneJ31":
                    self.vehicle_counter(vehicle_wait_time,road_id,["E15","-E0","eF_E",""]) #North,East,West,South
            

        state_space = [self.wait_time_N, self.wait_time_E, self.wait_time_W, self.wait_time_S,
                 self.no_veh_N, self.no_veh_E, self.no_veh_W, self.no_veh_S]
        
        return np.array(state_space)

    def choose_action(self, observation):
       
        if np.random.random() > self.epsilon:
            if self.no_veh_E > 0 and all([self.no_veh_N == 0, self.no_veh_S == 0]):
                action = 1  # Verde per Sud
            elif self.no_veh_W > 0 and all([self.no_veh_N == 0, self.no_veh_S == 0]):
                action = 1  # Verde per Sud
            elif self.no_veh_N > 0 and all([self.no_veh_E == 0, self.no_veh_W == 0]):
                # print(self.no_veh_N," ",self.TLC_name)
                action = 0  # Verde per Nord
            else:
                state = T.tensor(observation,dtype=T.float).to(self.q_eval.device)
                actions = self.q_eval.forward(state)
                action = T.argmax(actions).item()
        else:
                action = np.random.choice(self.action_space)
                

        self.reset_lane_traffic_info_params()

        return action
    # def choose_action(self, observation):
    #     # if self.no_veh_S > 0 and (self.no_veh_S > 0.5 * (self.no_veh_N + self.no_veh_E + self.no_veh_W)):
    #     #     action = 3
    #     # elif self.no_veh_N > 0 and (self.no_veh_N > 0.5 * (self.no_veh_S + self.no_veh_E + self.no_veh_W)):
    #     #     action = 4
    #     # elif self.no_veh_W > 0 and (self.no_veh_W > 0.5 * (self.no_veh_S + self.no_veh_E + self.no_veh_N)):
    #     #     action = 5
    #     # elif 
         
    #     if np.random.random() > self.epsilon:
    #             state = T.tensor(observation,dtype=T.float).to(self.q_eval.device)
    #             actions = self.q_eval.forward(state)
    #             action = T.argmax(actions).item()
    #     else:
    #             action = np.random.choice(self.action_space)

    #     self.reset_lane_traffic_info_params()

    #     return action

    def reset_lane_traffic_info_params(self):
        # Accumulated traffic queues in each direction
        self.no_veh_N = 0
        self.no_veh_E = 0
        self.no_veh_W = 0
        self.no_veh_S = 0

        # Accumulated wait time in each direction
        self.wait_time_N = 0
        self.wait_time_E = 0
        self.wait_time_W = 0
        self.wait_time_S = 0

    def decrement_epsilon(self):

        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min
                        

    def store_transition(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def sample_memory(self):
        state, action, reward, new_state = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_ = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def get_edges(self):
        edges_list = list()

        if self.TLC_name == "gneJ26":
            edges_list = ["eI_B","eI_D","eI_H","eI_F"] #North,East,West,South
        elif self.TLC_name == "gneJ27":
            edges_list = ["eH_A","eH_I","","eH_G"] #North,East,West,South
        elif self.TLC_name == "gneJ28":
            edges_list = ["eD_C","","eD_I","eD_E"] #North,East,West,South
        elif self.TLC_name == "gneJ29":
            edges_list = ["","eB_C","eB_A","eB_I"] #North,East,West,South
        elif self.TLC_name == "gneJ30":
            edges_list = ["eF_I","eF_E","eF_G",""] #North,East,West,South
        
        return edges_list
    
    #NO IDEA se sia giust. Di sicuro gli edges non lo sono
    def get_throughput(self):
        throughput = 0
        # Conta i veicoli che sono passati attraverso gli outgoing edges
        outgoing_edges = self.get_edges()  # Cambia con gli edge effettivi della tua rete
        for edge in outgoing_edges:
            if edge is not "":
                throughput += traci.edge.getLastStepVehicleNumber(edge)
        return throughput
    ### MAS
    def get_queue_length(self):
        return sum([self.no_veh_N, self.no_veh_E, self.no_veh_W, self.no_veh_S])
    def update_with_neighbor_info(self, neighbor_queues):
        self.neighbor_queues = neighbor_queues
    def print_neighbor_info(self, neighbor_queues):
        print(f"Agent {self.TLC_name} updating with neighbor queues: {neighbor_queues}")

    def printStatusReport(self, step):
        # Print status report
        phase_central = traci.trafficlight.getRedYellowGreenState(self.TLC_name)

        print("--- Status Report ---")
        print("Step: ", step)
        print("Signal State: ", phase_central)
        print("Last switch time at action: ", self.switch_time)
        print("Get next switch: ", (-self.switch_time + traci.trafficlight.getNextSwitch(self.TLC_name)))
        print("Get phase duration: ", (-self.switch_time + traci.trafficlight.getPhaseDuration(self.TLC_name)))


        print("no_veh_N: ", self.no_veh_N)
        print("no_veh_E: ", self.no_veh_E)
        print("no_veh_W: ", self.no_veh_W)
        print("no_veh_S: ", self.no_veh_S)

        print("wait_time_N", self.wait_time_N)
        print("wait_time_E", self.wait_time_E)
        print("wait_time_W", self.wait_time_W)
        print("wait_time_S", self.wait_time_S)

