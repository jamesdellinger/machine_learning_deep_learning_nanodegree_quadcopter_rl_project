import numpy as np
from physics_sim import PhysicsSim

class Custom_Quadcopter_Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

#         Goal: the location of the customer's house. z-position of the goal is 0 
#         because we want the copter to descend to ground level (in order to deliver 
#         the Amazon package to the customer's house).
        # Hover attempt is at an altitude of 100.
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 100.])
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        # Begin with a simple task: reward for hovering at height 100 (pose z-coordinate should be 100
        # and pose x and y coordinates should stay constant).
        
        
#         # Task is for the copter to take off from its base (initial location), and fly above a 
#         # minimum altitude to another location, and then land at that location, coming to a 
#         # complete rest. This task is inspired by the scenario of a package delivery drone 
#         # flying from Amazon's warehouse to delivery a package at a customer's house.
#         #
#         # Some important conditions to consider when modeling this task's reward:
#         # 1. Drone begins at rest, on the ground at the warehouse.
#         # 2. Drone should take most direct path to customer's house.
#         # 3. Drone should arrive at customer's house in shortest time possible.
#         # 4. Drone should fly at or above a minimum altitude, unless taking off or landing. 
#         # 5. When taking off, drone should be enticed to rise to minimum flying altitude of 100.
#         # 6. When landing at the customer's house, drone should be enticed to descend to 
#         #    ground and come to a rest.
#         #
#         # This task was purposefully conceived to solve the problem of making a one-way trip 
#         # from the warehouse to the customer's house. Once a copter drone had adequately 
#         # learned to make a one-way trip, the second half of a roundtrip is obviously just a 
#         # second one-way trip, with the simple modification of making the coordinates of the 
#         # warehouse's location the drone's new destination target.

#         reward = 0
#         # Entice the drone to take off from its initial location, rise to 100 in altitude, and 
#         # stay at or above 100 in altitude unless descending to the goal's location. 
#         #
#         # Unless the drone is above the x-y map coordinates of the customer's house, 
#         # penalize the drone for flying below the minimum required flight altitude. 
#         # This penalty descreases proportionally 
#         min_altitude = 100                         # required minimum flying altitude for drone 
#         drone_altitude = self.sim.pose[2]          # drone's current flying altitude
#         drone_map_loc = self.sim.pose[0:2]         # drone's current map location in x-y coordinates
#         customer_map_loc = self.target_pos[0:2]    # goal's map location in x-y coordinates
#         if np.less(drone_altitude, min_altitude).all() and np.not_equal(drone_map_loc, customer_map_loc).all():
#             reward -= (min_altitude-drone_altitude)/min_altitude
            
#         # Entice the drone to reach the x-y map location of the customer's 
#         # house as quickly as possible.
#         reward -= .3*(abs(drone_map_loc - customer_map_loc)).sum()
        
#         # Entice the drone to descend to the ground at the x-y map location of the 
#         # customer's house, and then come to a complete rest (all velocities 
#         # decrease to zero).
#         #
#         # Drone gets a greater reward as it descends further below the minimum 
#         # flying altitude, if the drone is above the customer's house. 
#         if np.array_equal(drone_map_loc, customer_map_loc):
#             reward += (min_altitude-drone_altitude)/min_altitude
#             if drone_altitude == 0:
#                 # Entice the drone to decrease its directional and angular 
#                 # velocities when it is on the ground at the customer's house, 
#                 # so that it can come to a complete rest. 
#                 #
#                 # At this point in the task, incremental rewards received as the agent 
#                 # slows its velocities must be greater than 1, which is the max reward 
#                 # that could be received by the agent decreasing its velocity to 0 -- 
#                 # we wouldn't want the agent to be incentivized to start hovering above 
#                 # the ground after it had fully descented to an altitude of 0.
#                 reward += 10/((self.sim.v + self.sim.anglular_v).sum() + 0.1)
                
#                 # Task is finished when drone is at customer's house's x-y coordinates, 
#                 # drone's altitude is 0, and drone's directional and angular velocities 
#                 # are all 0. When this state is reached, reward the drone 1000 for completing 
#                 # the task. (The reward chosen for this step needed to be at least greater than 100, 
#                 # which is the maximum reward that the drone could earn by decreasing its velocities 
#                 # toward 0 in the step just above.
#                 if self.sim.v == 0 and self.sim.anglular_v == 0:
#                     reward += 1000
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # Episode is done if goal is reached, or if the simulator informs that 
            # the copter has travelled outside of the bounds of the simulation or that 
            # the time limit has been exceeded.
            if np.array_equal(self.sim.pose[0:2], self.target_pos) and self.sim.v == 0 and self.sim.anglular_v == 0:
                done = True
            else:
                done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state