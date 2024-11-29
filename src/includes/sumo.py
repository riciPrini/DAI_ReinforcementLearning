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
             self.sumoCmd = ([sumoBinary, "-c", f"src/includes/sumo/2x2_compare/main.sumocfg",
                             "--tripinfo-output", "tripinfo.xml", "-S"])
        else:
            self.sumoCmd = ([sumoBinary, "-c", f"src/includes/sumo/{self.args.env}/main.sumocfg",
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
    def getVehicleControlTLS(self,vehicle_id):
        """
        Determina il semaforo che controlla il veicolo specificato.

        Args:
            env: Oggetto SUMO con l'API.
            vehicle_id: ID del veicolo.

        Returns:
            tls_id: ID del semaforo che controlla il veicolo (o None se non controllato).
        """
        try:
        # Ottieni l'ID della corsia in cui si trova il veicolo
            lane_id = traci.vehicle.getRoadID(vehicle_id)
        
        # Ottieni il semaforo che controlla la corsia
            tls_id = None
            if lane_id in ["eB_I","eD_I","eH_I","eF_I"]:
                tls_id = "gneJ26"
            elif lane_id in ["eA_H","eI_H","-E9","eG_H"]:
                tls_id = "gneJ27"
            elif lane_id in ["eC_D","-E16","eI_D","eE_D"]:
                tls_id = "gneJ28"
            elif lane_id in ["E4","eC_B","eA_B","eI_B"]:
                tls_id = "gneJ29"
            elif lane_id in ["eI_F","eE_F","eG_F",""]:
                tls_id = "gneJ30"

        
            return tls_id if tls_id else None
        except Exception as e:
            print(f"Errore nel determinare il TLS per il veicolo {vehicle_id}: {e}")
        return None