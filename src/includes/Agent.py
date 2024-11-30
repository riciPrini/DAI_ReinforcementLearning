import traci
import statistics as stat



class Agent:

    def __init__(self,TLC_name) :
        self.TLC_name = TLC_name


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
    def info(self):
        info = {
            "avg_wait_time" : self.get_avg_wait_time()
        }
        return info


