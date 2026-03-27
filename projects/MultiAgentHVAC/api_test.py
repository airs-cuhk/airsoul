import os
import sys
import numpy
import pickle
from copy import deepcopy

from airsoul.models import OmniRL_MultiAgent
from airsoul.utils import APIRunner
from airsoul.utils.tools import Configure
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hvac_api import HVACAPI

# env 相关
from data.anyhvac import create_cooler_cooler_graph, create_cooler_sensor_graph
from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
from env_wapper import HVACEnvWrapper, plot_cooler_values


def task_sampler_anyhvacv2(env, task):
        env.set_task(task)

        knn = 4
        obs_graph = create_cooler_sensor_graph(env, knn)
        agent_graph = create_cooler_cooler_graph(env, knn)
        agent_num, sensor_num = obs_graph.shape
        related_sensor = numpy.zeros((agent_num, knn), dtype=numpy.int32)
        related_agent = numpy.zeros((agent_num, knn), dtype=numpy.int32)
        for i in range(agent_num):
            sensor_indices = numpy.where(obs_graph[i] > 0)[0]
            sensor_weights = obs_graph[i][sensor_indices]
            sensor_sorted_indices = numpy.argsort(sensor_weights)
            sensor_indices = sensor_indices[sensor_sorted_indices]
            agent_indices = numpy.where(agent_graph[i] > 0)[0]
            agent_weights = agent_graph[i][agent_indices]
            agent_sorted_indices = numpy.argsort(agent_weights)
            agent_indices = agent_indices[agent_sorted_indices]
            related_sensor[i] = sensor_indices
            related_agent[i] = agent_indices

        return related_sensor, related_agent

def convert_env_action(env):
    action = deepcopy(env.current_action)
    action["value"] = env._action_value_to_temp(action["value"])
    return action

def build_up_env_action(action, previous_action):
    action = action - previous_action["value"]
    action_temp_diff_normalized = (action + 3) / 6
    return action_temp_diff_normalized

if __name__ == "__main__":
    # 1. 加载配置
    config = Configure()
    config.from_yaml(sys.argv[1])
    # 2. 创建 API Runner
    runner = APIRunner(config, use_gpu=True, world_size=1)
    # runner.reset_memory()
    # 3. 启动 Runner
    generator = runner.start(OmniRL_MultiAgent, HVACAPI)

    # 4. init env
    env = HVACEnvWrapper(reward_mode = config.generator_config.default_prompt, verbose=True)
    with open(config.generator_config.task_file, 'rb') as fr:
        task = pickle.load(fr)
    related_sensor, related_agent = task_sampler_anyhvacv2(env, task)

    # 5. init generator
    generator.init(env.target_temperature, related_sensor, related_agent)
    
    # 6. 开始交互
    previous_state = env.reset()[0]

    done = False
    while not done:
        previous_action = convert_env_action(env)
        action, vocab_seq_batch = generator.get_action(previous_state, previous_action)
        env_action = build_up_env_action(action, previous_action)
        state, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated
        
        current_action = convert_env_action(env)
        generator.icl(vocab_seq_batch, current_action, previous_action, done, reward=None)
        previous_state = state
        