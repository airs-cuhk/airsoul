import numpy as np

from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnv

class HVACEnvWrapper(HVACEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _create_agent_target_temperture(self, cooler_sensor_graph):
        n_coolers, n_sensors = self.cooler_sensor_topology.shape
        target_temp = self.target_temperature

        if isinstance(target_temp, (list, np.ndarray)):
            target_temps = np.array(target_temp)
        else:
            target_temps = np.full(self.n_sensors, target_temp)
        
        agent_target_temp = np.zeros(n_coolers, dtype=np.float32)
        for cooler_idx in range(n_coolers):
            connected_sensors = np.where(cooler_sensor_graph[cooler_idx] == 1.0)[0]
            if len(connected_sensors) > 0:
                avg_target_temp = np.mean(target_temps[connected_sensors])
            else:
                avg_target_temp = np.mean(target_temps)
            agent_target_temp[cooler_idx] = avg_target_temp
        self.agent_target_temp = agent_target_temp
        return
    
    def _compute_temperature_deviation(self, obs):

        if isinstance(obs, tuple):
            sensor_readings = obs[0][:len(self.sensors)]
        else:
            sensor_readings = obs[:len(self.sensors)]
        
        target_temp = self.target_temperature
        if isinstance(target_temp, (list, np.ndarray)):
            temperature_deviation = sensor_readings - np.array(target_temp)
        else:
            temperature_deviation = sensor_readings - target_temp
        
        return temperature_deviation