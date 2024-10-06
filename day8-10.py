import gym
import pybullet as p
import pybullet_data
import numpy as np
import time  # Used to slow down the simulation for visualization

class ReachAvoidEnv(gym.Env):
    def __init__(self, region_size_x=1.0, region_size_y=0.8, target_shape="line", target_size=0.1, max_time_steps=1000, attacker_start_position=None, attacker_velocity=0.0):
        # Connect to PyBullet in GUI mode for visualization
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Define play region size
        self.play_region_size_x = region_size_x
        self.play_region_size_y = region_size_y

        # Set target region parameters
        self.target_shape = target_shape  # "line" or "circle"
        self.target_size = target_size  # Size of the target region

        # Set max time steps (time limit)
        self.max_time_steps = max_time_steps

        # For "line" target, target_line_y defines the y-coordinate of the line
        self.target_line_y = 0.8
        # For "circle" target, target_position defines the center of the circle
        self.target_position = np.array([0.5, 0.9])

        # Create a 2D plane as the ground
        p.loadURDF("plane.urdf")

        # Create the boundaries of the play region (walls)
        self.create_boundaries()

        # Add static and dynamic obstacles
        self.static_obstacles = self.create_static_obstacles()
        self.dynamic_obstacles = self.create_dynamic_obstacles()

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # [steering, acceleration]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        # Initialize attacker's position and velocity
        if attacker_start_position is None:
            self.attacker_position = np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y]  # Random position if not specified
        else:
            self.attacker_position = np.clip(attacker_start_position, [0, 0], [self.play_region_size_x, self.play_region_size_y])  # Clip within region size

        self.attacker_velocity = attacker_velocity  # Set initial velocity
        self.attacker_heading = np.random.uniform(0, 2 * np.pi)  # Random heading angle (initial direction)
        self.attacker_max_speed = 0.05  # Maximum speed for the attacker
        self.attacker_turn_rate = 0.1  # Maximum turn rate (steering angle)

        # Initialize two defenders
        self.defender_positions = [
            np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y],
            np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y]
        ]
        self.time_step = 0

        # Initialize visual elements
        self.init_visual_elements()

    def create_boundaries(self):
        """Create boundaries (walls) around the play region to prevent agents from leaving the region."""
        wall_thickness = 0.02  # Reduced thickness to make boundaries smaller and more visible
        wall_height = 0.1  # Reduced height to make boundaries visible

        # Create walls (four sides) as thin boxes, adjusted to be within the play region
        self.boundaries = [
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.play_region_size_x / 2, wall_thickness, wall_height]),
                              basePosition=[self.play_region_size_x / 2, 0, 0]),  # Bottom wall
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.play_region_size_x / 2, wall_thickness, wall_height]),
                              basePosition=[self.play_region_size_x / 2, self.play_region_size_y, 0]),  # Top wall
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, self.play_region_size_y / 2, wall_height]),
                              basePosition=[0, self.play_region_size_y / 2, 0]),  # Left wall
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, self.play_region_size_y / 2, wall_height]),
                              basePosition=[self.play_region_size_x, self.play_region_size_y / 2, 0])  # Right wall
        ]

    def create_static_obstacles(self):
        """Create static obstacles that remain fixed in the environment."""
        # Static obstacles: Using boxes for better visibility
        static_obstacles = [
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05]),
                              basePosition=[self.play_region_size_x * 0.5, self.play_region_size_y * 0.5, 0]),  # Box in the center
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05]),
                              basePosition=[self.play_region_size_x * 0.7, self.play_region_size_y * 0.3, 0])  # Another box
        ]
        return static_obstacles

    def create_dynamic_obstacles(self):
        """Create dynamic obstacles that move around the environment."""
        # Dynamic obstacle: Using a sphere that moves in a visible way
        dynamic_obstacles = [
            p.createMultiBody(baseMass=1, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1]),
                              basePosition=[0.3, 0.3, 0])
        ]
        return dynamic_obstacles

    def move_dynamic_obstacles(self):
        """Move the dynamic obstacles to simulate real-world conditions."""
        # Example: Simple oscillatory motion for a dynamic obstacle
        for i, obstacle in enumerate(self.dynamic_obstacles):
            position = p.getBasePositionAndOrientation(obstacle)[0]
            new_x = position[0] + 0.01 * np.sin(self.time_step / 20)
            new_y = position[1] + 0.01 * np.cos(self.time_step / 20)
            p.resetBasePositionAndOrientation(obstacle, [new_x, new_y, 0], [0, 0, 0, 1])

    def init_visual_elements(self):
        """Initialize the visual elements for the attacker, defenders, and target."""
        # Add the attacker as a green visual sphere
        attacker_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
        self.attacker_id = p.createMultiBody(baseMass=1, baseVisualShapeIndex=attacker_visual_shape_id,
                                             basePosition=[self.attacker_position[0], self.attacker_position[1], 0])

        # Add defenders as red visual spheres (TWO defenders)
        self.defender_ids = []
        for pos in self.defender_positions:
            defender_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
            defender_id = p.createMultiBody(baseMass=1, baseVisualShapeIndex=defender_visual_shape_id,
                                            basePosition=[pos[0], pos[1], 0])
            self.defender_ids.append(defender_id)

        # Add the target based on the target shape (line or circle)
        if self.target_shape == "line":
            # A line is represented as a rectangle in PyBullet
            target_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.play_region_size_x / 2, 0.01, 0.01],
                                                         rgbaColor=[1, 1, 0, 1])
            self.target_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual_shape_id,
                                               basePosition=[self.play_region_size_x / 2, self.target_line_y, 0])
        elif self.target_shape == "circle":
            # A circle is represented as a small sphere
            target_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.target_size, rgbaColor=[1, 1, 0, 1])
            self.target_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual_shape_id,
                                               basePosition=[self.target_position[0], self.target_position[1], 0])

    def update_visual_elements(self):
        """Update the visual positions of the attacker and defenders."""
        p.resetBasePositionAndOrientation(self.attacker_id, [self.attacker_position[0], self.attacker_position[1], 0], [0, 0, 0, 1])
        for defender_id, pos in zip(self.defender_ids, self.defender_positions):
            p.resetBasePositionAndOrientation(defender_id, [pos[0], pos[1], 0], [0, 0, 0, 1])

    def reset(self):
        """Reset the environment to the initial state."""
        self.attacker_position = np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y]
        self.attacker_velocity = 0.0  # Reset velocity to zero
        self.attacker_heading = np.random.uniform(0, 2 * np.pi)  # Random initial heading
        self.defender_positions = [
            np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y],
            np.random.rand(2) * [self.play_region_size_x, self.play_region_size_y]
        ]
        self.time_step = 0
        self.update_visual_elements()  # Update the visual elements to reflect the reset
        return np.concatenate([self.attacker_position, *self.defender_positions, self.target_position])

    def step(self, action):
        """Execute the agent's action and update the environment."""
        # Unpack the action into steering (angle) and acceleration
        steering = action[0] * self.attacker_turn_rate  # Change in heading
        acceleration = action[1] * self.attacker_max_speed  # Change in speed

        # Update the heading (direction) and velocity (speed) based on the action
        self.attacker_heading += steering
        self.attacker_velocity = np.clip(self.attacker_velocity + acceleration, 0, self.attacker_max_speed)

        # Update the attacker's position based on its velocity and heading
        new_position_x = self.attacker_position[0] + self.attacker_velocity * np.cos(self.attacker_heading)
        new_position_y = self.attacker_position[1] + self.attacker_velocity * np.sin(self.attacker_heading)

        # Boundary check: If the attacker hits the boundary, adjust the heading
        if new_position_x <= 0 or new_position_x >= self.play_region_size_x:
            self.attacker_heading = np.pi - self.attacker_heading  # Reflect angle along the x-axis
        if new_position_y <= 0 or new_position_y >= self.play_region_size_y:
            self.attacker_heading = -self.attacker_heading  # Reflect angle along the y-axis

        # Update position after heading adjustment
        self.attacker_position[0] = np.clip(new_position_x, 0, self.play_region_size_x)
        self.attacker_position[1] = np.clip(new_position_y, 0, self.play_region_size_y)

        # Move dynamic obstacles
        self.move_dynamic_obstacles()

        # Update positions visually
        self.update_visual_elements()

        # Calculate the reward and check for the target condition based on the target shape
        if self.target_shape == "line":
            # For a line target, check if the attacker has crossed the target line (y-coordinate)
            distance_to_target_line = self.target_line_y - self.attacker_position[1]
            reward = -distance_to_target_line  # Reward decreases with distance to the target line
            done = self.attacker_position[1] >= self.target_line_y
        elif self.target_shape == "circle":
            # For a circular target, check if the attacker is within the target radius
            distance_to_target_center = np.linalg.norm(self.attacker_position - self.target_position)
            reward = -distance_to_target_center  # Reward decreases with distance to the target center
            done = distance_to_target_center <= self.target_size

        # Terminate the simulation after a maximum time step (time limit)
        self.time_step += 1
        if self.time_step >= self.max_time_steps:
            done = True
        
        return np.concatenate([self.attacker_position, *self.defender_positions, self.target_position]), reward, done, {}

    def set_attacker_attributes(self, position=None, velocity=None):
        """Set the attacker's starting position and velocity."""
        if position is not None:
            self.attacker_position = np.clip(position, [0, 0], [self.play_region_size_x, self.play_region_size_y])
        if velocity is not None:
            self.attacker_velocity = velocity

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

# Instantiate and run the environment with adjusted boundaries and obstacles
env = ReachAvoidEnv(region_size_x=2.0, region_size_y=1.5, target_shape="circle", target_size=0.2, max_time_steps=500)
env.set_attacker_attributes(position=[0.1, 0.1], velocity=0.02)  # Setting initial position and velocity of attacker
env.visualize()
env.close()
