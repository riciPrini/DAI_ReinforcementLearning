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
    
    def __init__(self,args):
        self.args = args
        # check binary
        if self.args.nogui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')
        
        self.sumoCmd = ([sumoBinary, "-c", f"includes/sumo/{self.args.env}/main.sumocfg",
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