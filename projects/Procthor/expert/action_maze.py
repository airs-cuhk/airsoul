import copy
import math
import debugpy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle  # Import Circle class
import random

AGENT_MOVEMENT_CONSTANT_L = 0.5  # Distance the robot moves each time
AGENT_MOVEMENT_CONSTANT_S = 0.25
AGENT_ROTATE_CONSTANT_L = 36
AGENT_ROTATE_CONSTANT_S = 9

MAX_SMALL_ROTATIONS = 5
MAX_LARGE_ROTATIONS = 6
SMALL_THRESHOLD = 0.5
# SMALL_THRESHOLD = 1e-3
# Define action constants
from collections import deque

MAX_ROTATION_ATTEMPTS = 5


ACTIONS = {
    'move_ahead': 'm',
    'move_ahead_small': 'ms',
    'rotate_left': 'l',
    'rotate_right': 'r',
    'rotate_left_small': 'ls',
    'rotate_right_small': 'rs',
    'end': 'end'
}


def sample_from_shortest_path(shortest_path, threshold):
    """
    Generate uniformly sampled points along the shortest path.

    :param shortest_path: List of points representing the shortest path
    :return: List of sampled points [(x1, y1), (x2, y2), ...]
    """

    def sample_line_segment(p1, p2):
        """
        Uniformly sample the line segment between two points.

        :param p1: First point (x1, y1)
        :param p2: Second point (x2, y2)
        :return: Sampled points [(x1, y1), (x2, y2), ...]
        """
        x1, y1 = p1['x'], p1['z']
        x2, y2 = p2['x'], p2['z']
        # Calculate the distance between two points
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if distance < threshold:
            return [(x1, y1), (x2, y2)]
        # Number of sample points
        num_samples = int(distance / threshold)

        # Calculate intervals in x and y directions
        x_interval = (x2 - x1) / num_samples
        y_interval = (y2 - y1) / num_samples

        # Generate uniformly sampled points
        samples = [(x1 + i * x_interval, y1 + i * y_interval) for i in range(num_samples + 1)]
        return samples

    points = []
    for i in range(len(shortest_path) - 1):
        start_p = shortest_path[i]
        end_p = shortest_path[i + 1]
        points.extend(sample_line_segment(start_p, end_p))

    return points


def check(point, shortest_path, threshold=0.25):
    """
    Check if a point is within a certain threshold distance from the shortest path.

    :param point: The point to check (x, y)
    :param shortest_path: List of points representing the shortest path
    :param threshold: Distance threshold
    :return: True if the point is within the threshold distance, False otherwise
    """
    points = sample_from_shortest_path(shortest_path, threshold)
    _points = list(points)
    distances = []

    if not _points:
        return True

    distances = [math.hypot(p[0] - point[0], p[1] - point[1]) for p in points]

    return min(distances) < threshold


def calculate_angle(current_pos, next_pos):
    """
    Calculate the angle from current_pos to next_pos, relative to the north (z-axis).

    :param current_pos: Current position, dictionary format {'x': float, 'z': float}
    :param next_pos: Next position, dictionary format {'x': float, 'z': float}
    :return: Angle in degrees, ranging from 0 to 360 degrees
    """
    dx = next_pos['x'] - current_pos['x']
    dz = next_pos['z'] - current_pos['z']
    return math.degrees(math.atan2(dx, dz)) % 360


def calculate_distance(current_pos, next_pos):
    """
    Calculate the Euclidean distance between current_pos and next_pos.

    :param current_pos: Current position, dictionary format {'x': float, 'z': float}
    :param next_pos: Next position, dictionary format {'x': float, 'z': float}
    :return: Distance in meters
    """
    dx = next_pos['x'] - current_pos['x']
    dz = next_pos['z'] - current_pos['z']
    return math.hypot(dx, dz)


def determine_turn_actions_(angle_diff):
    """
    Generate the required rotation actions based on the angle difference between the current heading and target heading.
    Use increments of 30° and 6° for rotation.

    :param angle_diff: Angle difference in degrees
    :return: (list of actions, updated heading)
    """
    actions = []
    small_rotation_count = 0
    large_rotation_count = 0
    # If the rotation reaches 180 degrees, perform a backward action

    # Large angle rotation 36°)
    while abs(angle_diff) >= AGENT_ROTATE_CONSTANT_L and large_rotation_count < MAX_LARGE_ROTATIONS:  # 6
        if angle_diff > 0:
            actions.append(ACTIONS['rotate_right'])
            angle_diff -= AGENT_ROTATE_CONSTANT_L
        else:
            actions.append(ACTIONS['rotate_left'])
            angle_diff += AGENT_ROTATE_CONSTANT_L
        large_rotation_count += 1

    # Small angle rotation (9°)
    while abs(angle_diff) >= AGENT_ROTATE_CONSTANT_S and small_rotation_count < MAX_SMALL_ROTATIONS:  # 2
        if angle_diff > 0:
            actions.append(ACTIONS['rotate_right_small'])
            angle_diff -= AGENT_ROTATE_CONSTANT_S
        else:
            actions.append(ACTIONS['rotate_left_small'])
            angle_diff += AGENT_ROTATE_CONSTANT_S
        small_rotation_count += 1

    return actions


def determine_turn_actions(current_heading, target_heading):
    """
    Generate the required rotation actions based on the current heading and target heading.
    Use increments of 36° and 9° for rotation.

    :param current_heading: Current heading in degrees
    :param target_heading: Target heading in degrees
    :return: (list of actions, updated heading)
    """
    angle_diff = (target_heading - current_heading + 360) % 360
    if angle_diff > 180:
        angle_diff -= 360  # Normalize angle difference to (-180, 180] range

    actions = []
    small_rotation_count = 0
    large_rotation_count = 0
    # Large angle rotation (36°)
    while abs(angle_diff) >= AGENT_ROTATE_CONSTANT_L and large_rotation_count < MAX_LARGE_ROTATIONS: # 6
        if angle_diff > 0:
            actions.append(ACTIONS['rotate_right'])
            current_heading = (current_heading + AGENT_ROTATE_CONSTANT_L) % 360
            angle_diff -= AGENT_ROTATE_CONSTANT_L
        else:
            actions.append(ACTIONS['rotate_left'])
            current_heading = (current_heading - AGENT_ROTATE_CONSTANT_L) % 360
            angle_diff += AGENT_ROTATE_CONSTANT_L
        large_rotation_count += 1
    # Small angle rotation (9°)
    # while abs(angle_diff) >= AGENT_ROTATE_CONSTANT_S and small_rotation_count < MAX_SMALL_ROTATIONS and random.random() < SMALL_THRESHOLD:  # 2
    while abs(angle_diff) >= AGENT_ROTATE_CONSTANT_S and small_rotation_count < MAX_SMALL_ROTATIONS: # 2
        if angle_diff > 0:
            actions.append(ACTIONS['rotate_right_small'])
            current_heading = (current_heading + AGENT_ROTATE_CONSTANT_S) % 360
            angle_diff -= AGENT_ROTATE_CONSTANT_S
        else:
            actions.append(ACTIONS['rotate_left_small'])
            current_heading = (current_heading - AGENT_ROTATE_CONSTANT_S) % 360
            angle_diff += AGENT_ROTATE_CONSTANT_S
        small_rotation_count += 1
    return actions, current_heading


def determine_forward_actions(current_heading, current_pose, distance, ):
    """
    Determine the number of move_ahead actions needed based on the distance, with each step length being 0.2 meters.

    :param distance: Total distance in meters
    :param step_length: Distance moved per step in meters (default is 0.2 meters)
    :return: List of actions
    """
    actions = []

    # Large movement (0.5)
    large_move_count = int(distance / AGENT_MOVEMENT_CONSTANT_L)
    actions.extend([ACTIONS['move_ahead']] * large_move_count)


    # small movement (0.25)
    remaining_distance = distance - (large_move_count * AGENT_MOVEMENT_CONSTANT_L)
    small_move_count = int(remaining_distance / AGENT_MOVEMENT_CONSTANT_S)
    actions.extend([ACTIONS['move_ahead_small']] * small_move_count)


    # Handle remaining distance
    remaining_distance = remaining_distance - (small_move_count * AGENT_MOVEMENT_CONSTANT_S)
    new_x = current_pose['x'] + (large_move_count * AGENT_MOVEMENT_CONSTANT_L + small_move_count * AGENT_MOVEMENT_CONSTANT_S) * math.sin(math.radians(current_heading))
    new_z = current_pose['z'] + (large_move_count * AGENT_MOVEMENT_CONSTANT_L + small_move_count * AGENT_MOVEMENT_CONSTANT_S) * math.cos(math.radians(current_heading))


    current_pose = {'x':new_x, 'z':new_z}
    # if remaining_distance >= (AGENT_MOVEMENT_CONSTANT_S / 2):
    #     actions.append(ACTIONS['move_ahead_small'])


    return actions, current_pose


def plot_grid_map(shortest_path, actions, initial_pose, step_length=AGENT_MOVEMENT_CONSTANT_S, save_path=None, threshold=0.25, reachable_positions=None):
    """
    Use Matplotlib to plot a grid map and visualize the robot's movement path and actions.

    :param shortest_path: List of shortest path points, each point in dictionary format {'x': float, 'y': float, 'z': float}
    :param actions: List of actions
    :param initial_pose: Initial pose, including position and rotation angle
    :param step_length: Step length for each movement in meters
    :param save_path: Path to save the plot
    :param threshold: Distance threshold for checking proximity to the shortest path
    """
    # if reachable_positions is not None:
    #     """reachable_positions"""
    #     xs = [rp["x"] for rp in reachable_positions]
    #     zs = [rp["z"] for rp in reachable_positions]
    #     # print(self.reachable_positions)

    #     _, ax = plt.subplots(figsize=(10, 10))
    #     ax.set_aspect('equal', 'box')
    #     xs = [rp["x"] for rp in reachable_positions]
    #     zs = [rp["z"] for rp in reachable_positions]
    #     # print(self.reachable_positions)
    #     ax.scatter(xs, zs)
    #     plt.savefig(f"{save_path}/reachable_positions.jpg")
        




    _, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', 'box')
    # Draw grid
    x_min = min(point['x'] for point in shortest_path) - 1
    x_max = max(point['x'] for point in shortest_path) + 1
    z_min = min(point['z'] for point in shortest_path) - 1
    z_max = max(point['z'] for point in shortest_path) + 1

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title('Robot Path')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Draw shortest path
    path_x = [point['x'] for point in shortest_path]
    path_z = [point['z'] for point in shortest_path]
    ax.plot(path_x, path_z, 'bo-', label='shortest path')

    # Execute actions and record poses
    pose_sequence = execute_actions(actions, initial_pose)

    # Draw robot poses and heading arrows
    for idx, pose in enumerate(pose_sequence):
        pos = pose['position']
        rot = pose['rotation']['y']
        ax.plot(pos['x'], pos['z'], 'ro')  # Draw robot position

        if not check((pos['x'], pos['z']), shortest_path):
            circle = Circle((pos['x'], pos['z']), radius=AGENT_MOVEMENT_CONSTANT_S, color='g', fill=True)
            ax.add_patch(circle)
        # Draw heading arrow
        arrow_length = 0.2  # Arrow length
        arrow_dx = arrow_length * math.sin(math.radians(rot))
        arrow_dz = arrow_length * math.cos(math.radians(rot))
        ax.arrow(pos['x'], pos['z'], arrow_dx, arrow_dz, head_width=0.1, head_length=0.1, fc='r', ec='r')

        # Circle sample
        circle = Circle((pos['x'], pos['z']), radius=AGENT_MOVEMENT_CONSTANT_S, color='r', fill=False)
        ax.add_patch(circle)

        # Annotate steps
        ax.text(pos['x'] + 0.1, pos['z'] + 0.1, f'{idx}', fontsize=8, color='green')

    if reachable_positions is not None:
        xs = [rp["x"] for rp in reachable_positions]
        zs = [rp["z"] for rp in reachable_positions]
        # print(self.reachable_positions)
        ax.scatter(xs, zs)
        plt.savefig(f"{save_path}/robot_path_full.jpg")
        
    
    x_min = min(point['x'] for point in shortest_path) - 1
    x_max = max(point['x'] for point in shortest_path) + 1
    z_min = min(point['z'] for point in shortest_path) - 1
    z_max = max(point['z'] for point in shortest_path) + 1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)

    ax.legend()
    plt.savefig(f"{save_path}/robot_path_region.jpg")
    plt.close()



def execute_actions(actions, initial_pose):
    """
    Update the robot's position coordinates (x, z) and orientation angle (y) based on the action sequence.

    :param actions: List of actions, e.g., ['move_ahead', 'rotate_right', ...]
    :param initial_pose: Initial pose, including position and rotation angle
                         Format:
                         {
                             'position': {'x': 0, 'z': 0},
                             'rotation': {'y': 0}  # In degrees
                         }
    :return: List of pose sequences, each element containing the current position and orientation
             Format:
             [
                 {'position': {'x': ..., 'z': ...}, 'rotation': {'y': ...}},
                 ...
             ]
    """
    pose_sequence = [copy.deepcopy(initial_pose)]
    current_pose = copy.deepcopy(initial_pose)  # Copy initial pose
    current_heading = current_pose['rotation']['y']  # Current heading angle
    # rotation_attempts = deque(maxlen=MAX_ROTATION_ATTEMPTS)  # Track the recent rotation actions

    for action in actions:
        # if action in [ACTIONS['rotate_left'], ACTIONS['rotate_right']]:
        #     rotation_attempts.append(action)

        # if len(rotation_attempts) == rotation_attempts.maxlen and len(set(rotation_attempts)) == 1:
        #     # If the robot has been continuously rotating, attempt to move back
        #     print("Too many consecutive rotations. Adding a move back action to escape.")
        #     action = ACTIONS['move_back']

        if action == ACTIONS['move_ahead']:
            # Move ahead 0.2 meters in the current heading
            radians = math.radians(current_heading)
            dx = AGENT_MOVEMENT_CONSTANT_L * math.sin(radians)
            dz = AGENT_MOVEMENT_CONSTANT_L * math.cos(radians)
            current_pose['position']['x'] += dx
            current_pose['position']['z'] += dz

        elif action == ACTIONS['move_ahead_small']:
            # Move back 0.2 meters in the current heading
            radians = math.radians(current_heading)
            dx = AGENT_MOVEMENT_CONSTANT_S * math.sin(radians)
            dz = AGENT_MOVEMENT_CONSTANT_S * math.cos(radians)
            current_pose['position']['x'] += dx
            current_pose['position']['z'] += dz

        elif action == ACTIONS['rotate_left']:
            # Rotate left 36 degrees
            current_heading = (current_heading - AGENT_ROTATE_CONSTANT_L) % 360
            current_pose['rotation']['y'] = current_heading

        elif action == ACTIONS['rotate_right']:
            # Rotate right 36 degrees
            current_heading = (current_heading + AGENT_ROTATE_CONSTANT_L) % 360
            current_pose['rotation']['y'] = current_heading

        elif action == ACTIONS['rotate_left_small']:
            # Rotate left 9 degrees
            current_heading = (current_heading - AGENT_ROTATE_CONSTANT_S) % 360
            current_pose['rotation']['y'] = current_heading

        elif action == ACTIONS['rotate_right_small']:
            # Rotate right 9 degrees
            current_heading = (current_heading + AGENT_ROTATE_CONSTANT_S) % 360
            current_pose['rotation']['y'] = current_heading

        elif action == ACTIONS['end']:
            # End task, stop executing actions
            break

        # Record current pose
        pose_sequence.append(copy.deepcopy(current_pose))  # Append a copy of current_pose

    return pose_sequence


def shortest_path_to_actions(shortest_path, initial_pose):
    """
    Convert the shortest path to a series of discrete actions.

    :param shortest_path: List of shortest path points, each point in dictionary format {'x': float, 'y': float, 'z': float}
    :param initial_pose: Initial pose, including position and rotation
                         Format:
                         {
                             'position': {'x': float, 'z': float},
                             'rotation': {'y': float}  # In degrees
                         }
    :return: List of actions
    """

    actions = []

    current_pose = {'x': initial_pose['position']['x'],
                    'z': initial_pose['position']['z']}
    current_heading = initial_pose['rotation']['y']

    for i in range(len(shortest_path)):
        if i==0:
            continue
        point = shortest_path[i]
        next_pos = {'x': point['x'], 'z': point['z']}

        # Calculate target angle

        target_angle = calculate_angle(current_pose, next_pos)

        # Generate rotation actions
        turn_actions, current_heading = determine_turn_actions(current_heading, target_angle)
        actions.extend(turn_actions)

        # Calculate moving distance
        distance = calculate_distance(current_pose, next_pos)

        # Generate forward actions(自转->前进)
        forward_actions, current_pose = determine_forward_actions(current_heading, current_pose, distance)
        actions.extend(forward_actions)

    actions.append(ACTIONS['end'])
    return actions


# Example usage
if __name__ == "__main__":
    # Define the shortest path
    shortest_path = [
        {'x': 8.933333396911621, 'y': 0.039272308349609375, 'z': 5.400000095367432},
         {'x': 9.59999942779541, 'y': 0.039272308349609375, 'z': 5.849999904632568},
         {'x': 10.050000190734863, 'y': 0.039272308349609375, 'z': 6.0},
         {'x': 9.866666793823242, 'y': 0.039272308349609375, 'z': 6.40000057220459},
         {'x': 9.466667175292969, 'y': 0.039272308349609375, 'z': 6.800000190734863},
         {'x': 8.933333396911621, 'y': 0.039272308349609375, 'z': 6.800000190734863},
         {'x': 7.649999618530273, 'y': 0.039272308349609375, 'z': 6.450000286102295},
         {'x': 6.800000190734863, 'y': 0.039272308349609375, 'z': 5.333333492279053},
         {'x': 6.149999618530273, 'y': 0.039272308349609375, 'z': 3.9000000953674316},
         {'x': 5.733333587646484, 'y': 0.039272308349609375, 'z': 3.7333335876464844},
         {'x': 4.400000095367432, 'y': 0.039272308349609375, 'z': 3.066666841506958},
         {'x': 3.066666841506958, 'y': 0.039272308349609375, 'z': 3.066666841506958},
         {'x': 1.8666667938232422, 'y': 0.039272308349609375, 'z': 3.7333335876464844},
         {'x': 1.6000001430511475, 'y': 0.039272308349609375, 'z': 4.0},
         {'x': 1.492241382598877, 'y': 0.039272308349609375, 'z': 5.831896781921387}
    ]

    # shortest_path = [
    #     {'x': 1, 'y': 0.039272308349609375, 'z': 2},
    #     {'x': 2, 'y': 0.039272308349609375, 'z': 5},
    #     {'x': 4, 'y': 0.039272308349609375, 'z': 6},
    # ]

    # Define the initial pose
    initial_pose = {
        'position': {'x': shortest_path[0]['x'], 'z': shortest_path[0]['z']},
        'rotation': {'y': 0}  # Initial heading is north (0 degrees)
    }

    # Convert the shortest path to a list of actions
    actions = shortest_path_to_actions(shortest_path, initial_pose)
    print("Generated action sequence:", actions)

    # Visualization
    plot_grid_map(shortest_path, actions, initial_pose, save_path="test-actions-maze.png", step_length=AGENT_MOVEMENT_CONSTANT_S)