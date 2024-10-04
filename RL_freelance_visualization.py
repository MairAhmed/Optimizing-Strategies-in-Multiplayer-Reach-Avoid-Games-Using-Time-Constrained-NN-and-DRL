import gym
import pybullet as p
import pybullet_data
import numpy as np
import time  # Used to slow down the simulation for visualization

class ReachAvoidEnv(gym.Env):
    def __init__(self, region_size_x=1.0, region_size_y=0.8):
        # Connect to PyBullet in GUI mode for visualization
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Define play region size
        self.play_region_size_x = region_size_x
        self.play_region_size_y = region_size_y
        self.target_region_y = 1
        self.target_line_y = 0.8

        # Create a 2D plane as the ground
        p.loadURDF("plane.urdf")

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        # Initialize attacker, defenders, and their positions
        self.attacker_position = np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y]
        self.defender_positions = [
            np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y],
            np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y]
        ]
        self.target_position = np.array([0.5, self.target_line_y + (self.target_region_y - self.target_line_y) / 2])
        self.time_step = 0

        # Initialize visual elements
        self.init_visual_elements()

    def init_visual_elements(self):
        """Initialize the visual elements for the attacker, defenders, and target."""
        # Add the attacker as a green visual sphere
        attacker_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
        self.attacker_id = p.createMultiBody(baseMass=1, baseVisualShapeIndex=attacker_visual_shape_id,
                                              basePosition=[self.attacker_position[0], self.attacker_position[1], 0])

        # Add defenders as red visual spheres
        self.defender_ids = []
        for pos in self.defender_positions:
            defender_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
            defender_id = p.createMultiBody(baseMass=1, baseVisualShapeIndex=defender_visual_shape_id,
                                             basePosition=[pos[0], pos[1], 0])
            self.defender_ids.append(defender_id)

        # Add the target as a yellow visual sphere for visual clarity
        target_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 1, 0, 1])
        self.target_id = p.createMultiBody(baseMass=1, baseVisualShapeIndex=target_visual_shape_id,
                                            basePosition=[self.target_position[0], self.target_position[1], 0])

    def update_visual_elements(self):
        """Update the visual positions of the attacker and defenders."""
        p.resetBasePositionAndOrientation(self.attacker_id, [self.attacker_position[0], self.attacker_position[1], 0], [0, 0, 0, 1])
        for defender_id, pos in zip(self.defender_ids, self.defender_positions):
            p.resetBasePositionAndOrientation(defender_id, [pos[0], pos[1], 0], [0, 0, 0, 1])

    def reset(self):
        """Reset the environment to the initial state."""
        self.attacker_position = np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y]
        self.defender_positions = [
            np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y],
            np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y]
        ]
        self.time_step = 0
        self.update_visual_elements()  # Update the visual elements to reflect the reset
        return np.concatenate([self.attacker_position, *self.defender_positions, self.target_position])

    def step(self, action):
        """Execute the agent's action and update the environment."""
        action = np.clip(action, -1, 1)
        self.attacker_position += action * 0.05  # Update the attacker position
        
        # Keep the attacker within the play region boundaries
        self.attacker_position = np.clip(self.attacker_position, [0, 0], [self.play_region_size_x, self.play_region_size_y])

        # Update positions visually
        self.update_visual_elements()

        # Calculate the distance to the target line
        distance_to_target_line = self.target_line_y - self.attacker_position[1]
        reward = -distance_to_target_line  # Reward decreases with distance to the target line

        # Check if the attacker has reached the target region
        done = False
        if self.attacker_position[1] >= self.target_line_y:
            reward = 100  # Big reward for reaching the target region
            done = True

        # Terminate the simulation after a maximum time step
        self.time_step += 1
        if self.time_step > 1000:
            done = True
        
        return np.concatenate([self.attacker_position, *self.defender_positions, self.target_position]), reward, done, {}

    def close(self):
        """Terminate the environment."""
        p.disconnect(self.physics_client)

    def visualize(self):
        """Run the simulation and visualize the environment in real-time until closed manually."""
        obs = self.reset()
        while True:
            action = self.action_space.sample()  # Sample a random action for testing
            obs, reward, done, info = self.step(action)

            p.stepSimulation()  # Ensure PyBullet steps the simulation forward
            time.sleep(0.05)    # Slow down the loop to make it viewable

            if done:
                break

            # Check if the user has closed the PyBullet window
            if p.getConnectionInfo(self.physics_client)['connectionMethod'] == p.GUI:
                if p.isConnected(self.physics_client) == 0:
                    break

# Instantiate and run the environment
env = ReachAvoidEnv(region_size_x=1.0, region_size_y=0.8)  # You can adjust these values for scalability
env.visualize()
env.close()
