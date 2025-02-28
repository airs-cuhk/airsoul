import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import imageio

class AgentVisualizer:
    def __init__(self, save_path, visualize_online=False, skip_episode=0, fig=None, ax=None):
        self.save_path = save_path
        self.visualize_online = visualize_online
        self.skip_episode = skip_episode

        self.G = nx.DiGraph()
        self.fixed_positions = {}
        self.total_reward = 0
        
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(figsize=(15, 7))
        else:
            self.fig = fig
            self.ax = ax

        self.episode = 0
        self.current_state = None 

    def step(self, last_state, action, reward, next_state, done):
        if not self.G.has_node(last_state):
            self.G.add_node(last_state)
        if not self.G.has_node(next_state):
            self.G.add_node(next_state)
        if not self.G.has_edge(last_state, next_state):
            self.G.add_edge(last_state, next_state, weight=reward)

        # Initialize positions for new nodes
        if last_state not in self.fixed_positions:
            pos = self._compute_new_position()
            self.fixed_positions[last_state] = pos

        if next_state not in self.fixed_positions:
            pos = self._compute_new_position()
            self.fixed_positions[next_state] = pos

        self.total_reward += reward
        
        if done:
            self.episode += (1 + self.skip_episode)
            self.total_reward = 0  

    def _compute_new_position(self):
        # Generate a new position and ensure it's at least 1.0 distance from all existing positions
        while True:
            pos = np.random.rand(2) * 20
            if all(np.linalg.norm(pos - p) >= 0.8 for p in self.fixed_positions.values()):
                return pos

    def update(self, frame):
        last_state, action, reward, next_state, done = frame
        self.step(last_state, action, reward, next_state, done)
        
        self.current_state = next_state

        pos = self.fixed_positions

        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in self.G.edges(data=True)}

        self.ax.clear()

        nx.draw_networkx_nodes(self.G, pos, node_size=700)
        nx.draw_networkx_edges(self.G, pos, arrowstyle='-')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        nx.draw_networkx_labels(self.G, pos, font_size=16, font_family="sans-serif")

        node_colors = ['orange' if n == self.current_state else 'skyblue' for n in self.G.nodes()]
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=700)

        plt.title(f"Anymdp (Episode {self.episode})")
        
        plt.text(0.02, 0.98, f'Episode Reward: {self.total_reward:.2f}', transform=self.ax.transAxes, fontsize=14,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
        
        plt.text(0.98, 0.98, f'Action: {action}', transform=self.ax.transAxes, fontsize=14,
                 verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
        
    def init(self):
        self.ax.clear()
        return []

    def draw(self, frames):
        if self.visualize_online:
            ani = FuncAnimation(self.fig, self.update, frames=frames, init_func=self.init, interval=100, blit=False, repeat=False)
            plt.show()  # Show the animation if visualizing
        else:
            imageio_writer = imageio.get_writer(f'{self.save_path}/anymdp.gif', mode='I', fps=10)
            for frame in frames:
                self.update(frame)  # Update the plot
                self.fig.canvas.draw_idle()  # Draw the canvas and flush it
                frame_data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                imageio_writer.append_data(frame_data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,)))
            imageio_writer.close()
        return