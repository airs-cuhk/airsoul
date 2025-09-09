import numpy as np
from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnv

class HVACEnvWrapper(HVACEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _create_cooler_sensor_graph(self, num_closest_sensors: int = 3):
        """
        Create sensor-cooler relationship graph.

        Args:
            num_closest_sensors (int): The number of closest sensors to consider for each cooler.
                                       Defaults to 3.

        Returns:
            np.ndarray: The sensor-cooler relationship graph.
        """
        # Create a new matrix with the same shape as the original (cooler, sensor) matrix
        obs_graph_orig = np.zeros_like(self.cooler_sensor_topology)
        n_coolers, n_sensors = self.cooler_sensor_topology.shape
        
        for cooler_id in range(n_coolers):
            sensor_weights = self.cooler_sensor_topology[cooler_id, :]
            if n_sensors >= num_closest_sensors:
                # Get the indices of the 'num_closest_sensors' smallest values (closest sensors)
                closest_sensor_idx = np.argpartition(sensor_weights, num_closest_sensors)[:num_closest_sensors]
            else:
                # If there are fewer than 'num_closest_sensors' sensors, use all sensors
                closest_sensor_idx = np.arange(len(sensor_weights))
            
            # Set the positions corresponding to the closest sensors to 1
            obs_graph_orig[cooler_id, closest_sensor_idx] = 1.0
        
        # Transpose the matrix to match the required shape (sensor, cooler)
        # obs_graph = obs_graph_orig.T.astype(np.float32)
        
        self.obs_graph = obs_graph_orig
        return obs_graph_orig

    def _create_cooler_cooler_graph(self, k_nearest_coolers: int = 3):
        """
        Create cooler-cooler relationship graph using KNN.

        Args:
            k_nearest_coolers (int): The number of nearest coolers to consider for each cooler.
                                     Defaults to 3.

        Returns:
            np.ndarray: The cooler-cooler relationship graph.
        """
        n_coolers = self.cooler_sensor_topology.shape[0]
        agent_graph = np.zeros((n_coolers, n_coolers), dtype=np.float32)
        
        # Get cooler positions
        # Assuming self.coolers is an iterable of objects with a 'loc' attribute
        cooler_positions = np.array([cooler.loc for cooler in self.coolers])
        
        # Compute pairwise distances
        for i in range(n_coolers):
            for j in range(n_coolers):
                if i != j:
                    dist = np.linalg.norm(cooler_positions[i] - cooler_positions[j])
                    agent_graph[i, j] = dist
        
        # Convert to KNN graph (k='k_nearest_coolers' nearest neighbors)
        k = min(k_nearest_coolers, n_coolers - 1)
        for i in range(n_coolers):
            # Get k nearest neighbors
            distances = agent_graph[i, :]
            nearest_indices = np.argsort(distances)[1:k+1]  # Exclude self
            
            # Set connections
            agent_graph[i, :] = 0
            agent_graph[i, nearest_indices] = 1
        
        # Make symmetric and add self-connections (if desired, currently commented out)
        # agent_graph = np.maximum(agent_graph, agent_graph.T)
        # np.fill_diagonal(agent_graph, 1.0) # Uncomment if self-connections are needed
        self.agent_graph = agent_graph
        return agent_graph
    
    def _create_agent_target_temperture(self):
        # call _create_cooler_cooler_graph() before _create_agent_target_temperture()
        n_coolers, n_sensors = self.cooler_sensor_topology.shape
        target_temp = self.target_temperature

        if isinstance(target_temp, (list, np.ndarray)):
            target_temps = np.array(target_temp)
        else:
            target_temps = np.full(self.n_sensors, target_temp)
        
        cooler_sensor_graph = self.obs_graph
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