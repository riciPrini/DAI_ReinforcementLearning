import os
import sys
import optparse 

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # Checks for the binary in environ vars
import traci

class SimEnv():
    """
    This class hendle SUMO simulation environment.
    """
    
    def __init__(self,args,compare=False):
        self.total_throughput = 0
        self.total_departed = 0
        self.args = args
        # check binary
        if self.args.nogui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')
        if compare:

             self.sumoCmd = ([sumoBinary, "--no-warnings","-c", f"src/includes/sumo/{self.args.type}/main.sumocfg",
                             "--tripinfo-output", "tripinfo.xml", "-S"])
        else:
            self.sumoCmd = ([sumoBinary, "--no-warnings", "-c", f"src/includes/sumo/{self.args.env}/main.sumocfg",
                             "--tripinfo-output", "tripinfo.xml", "-S"])


    # START SIMULATION
    def start_sumo(self):
        """
        Start sumo simulation.
        """
        traci.start(self.sumoCmd)

    # END SIMULATION
    def close_sumo(self):
        """
        End sumo simulation.
        """
        traci.close()

    # SIMULATION STEP
    def simulationStep(self):
        """
        Increas Simulation one step.
        """
        traci.simulationStep()

    def getIDList(self):
        return traci.vehicle.getIDList()
    
    def getAccumulatedWaitingTime(self,id):
        return traci.vehicle.getAccumulatedWaitingTime(id)
        
    def getSpeed(self,id):
        return traci.vehicle.getSpeed(id)

    def getArrivedVehicles(self):
        return self.total_throughput
    
    def getDepartedVehicles(self):
        return self.total_departed
    
    def ArrivedVehicles(self):

        arrived = traci.simulation.getArrivedNumber()
        departed = traci.simulation.getDepartedNumber()
        vehicles_in_network = traci.vehicle.getIDList()

        self.total_throughput += arrived
        self.total_departed += departed

        # print(f"Arrived: {arrived}, Departed: {departed}, In Network: {len(vehicles_in_network)}")
    