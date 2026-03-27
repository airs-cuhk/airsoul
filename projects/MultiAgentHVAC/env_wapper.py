import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from xenoverse.anyhvac.anyhvac_env import HVACEnvDiffAction

class HVACEnvWrapper(HVACEnvDiffAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _create_agent_target_temperture(self, cooler_sensor_graph):
        n_coolers, n_sensors = self.cooler_sensor_topology.shape
        target_temp = self.target_temperature

        if isinstance(target_temp, (list, np.ndarray)):
            target_temps = np.array(target_temp)
        else:
            target_temps = np.full(n_sensors, target_temp)
        
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

        sensor_readings = obs["sensor_readings"]
        
        target_temp = self.target_temperature
        if isinstance(target_temp, (list, np.ndarray)):
            temperature_deviation = sensor_readings - np.array(target_temp)
        else:
            temperature_deviation = sensor_readings - target_temp
        
        return temperature_deviation
    
    def _convert_env_action(self, env_action):
        target_temp = self.target_temperature
        temp_value = self._action_value_to_temp(env_action["value"]) - target_temp
        switch = env_action["switch"]
        data = np.column_stack((temp_value, switch))
        result = data.reshape((len(temp_value), 1, 2))
        return result
    
def plot_cooler_values(values, output_dir, output_name, n_coolers, show_plot=False):
    """
    Draw and save the numerical curve graphs of multiple coolers
    
    Params:
    values (list or np.array): A two-dimensional array with the shape of (time steps, number of coolers)
    output_dir (str): Output file save directory
    n_coolers (int): The number of coolers

    show_plot (bool)
    
    Return:
    tuple: (csv file path, figure path)
    """
    os.makedirs(output_dir, exist_ok=True)
    values_array = np.array(values)
    df = pd.DataFrame(values_array, columns=[f"cooler_{i}" for i in range(n_coolers)])
    csv_path = os.path.join(output_dir, "cooler_values.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data save to: {csv_path}")
    
    min_width_per_plot = 4
    min_height_per_plot = 2
    max_width = 24
    max_height = 36
    
    n_cols = min(4, n_coolers)
    n_rows = (n_coolers + n_cols - 1) // n_cols
    
    min_width_per_plot *= 4
    max_width *= 4
    
    fig_width = min(n_cols * min_width_per_plot, max_width)
    fig_height = min(n_rows * min_height_per_plot, max_height)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=0.3,
        hspace=0.4
    )
    
    for i in range(n_coolers):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        cooler_values = values_array[:, i]
        
        ax.plot(cooler_values, 'b-', linewidth=0.5)
        
        ax.set_title(f"Cooler {i}", fontsize=10)
        ax.set_xlabel("Time Step", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    plot_path = os.path.join(output_dir, f"{output_name}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Figure save to: {plot_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()
    
    return csv_path, plot_path