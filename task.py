import numpy as np
import math
from physics_sim import PhysicsSim

class Custom_Task():
    
    """Task (environment) that defines the goal and provides feedback to the agent.
    
       This task will be a straightforward takeoff task: the quadcopter begins 
       the task at rest on the ground at the center of the map -- at x-y-z coordinates 
       of (0,0,0). Because the copter begins at rest, its initial velocities 
       and its initial angular velocities are both 0.
       
       Once the task commences, the quadcopter must takeoff and elevate to an altitude 
       of 10 meters above the ground as rapidly as possible. It should do this while 
       maintaining its location above the center of the x-y plane. 
       
       When the quadcopter reaches a height of 10 meters, it should continue hovering 
       at this height, and at a position directly above the center of the map at x-y 
       coordinates of (0,0). 
       
       The quadcopter's target position will thus be (0,0,10).
    """
    
    def __init__(self, init_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), init_velocities=np.array([0.0, 0.0, 0.0]), 
        init_angle_velocities=np.array([0.0, 0.0, 0.0]), runtime=5., target_pos=np.array([0., 0., 10.])):
        
        """Initialize a Task object.
        Params
        ======
            init_pose: Initial position of the quadcopter in (x,y,z) dimensions and the Euler angles.
                       For this custom task, since starting location is the ground at the center of 
                       the x-y plane, we set init_pose to np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).
                       
            init_velocities: Initial velocity of the quadcopter in (x,y,z) dimensions.
                             This custom task begins with the copter at rest, so we set 
                             init_velocities to np.array([0.0, 0.0, 0.0]).
                             
            init_angle_velocities: Initial radians/second for each of the three Euler angles. 
                                   This custom task begins with the copter at rest, so we also set 
                                   init_angle_velocities to np.array([0.0, 0.0, 0.0]).
            
            runtime: Time limit for each episode. Set this large enough to give the agent time to 
                     reach the target position. Setting runtime to 5. gives the agent enough time.
            
            target_pos: Target/goal (x,y,z) position for the agent. This task's goal is for the 
                        quadcopter to reach a height of 10 meters above the x-y plane's center, 
                        so target_pos is set to np.array([0., 0., 10.]).
        """
        
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        # State size for this task contains only x-y-z coordinates.
        self.state_size = self.action_repeat * 3
        
        # Action_low must be greater than 0, so that physics_sim.py doesn't 
        # throw a divide by zero error when calculating the copter's thrust. 
        # For this take-off task, keeping the minimum rotor speed relatively 
        # high (at 800) is the only way I can ensure that the copter learns 
        # to take off.
        self.action_low = 800
        self.action_high = 900
        self.action_size = 4
        
        # Goal is a hovering position of (0,0,10)
        self.target_pos = target_pos 
        
    def get_reward(self):
        
        """Uses current pose of sim to return reward."""
        
        current_position = self.sim.pose[:3]
        target_position = self.target_pos[:3]
        
        # Penalize the copter when it is far away from its target position. 
        # The closer the copter gets to its target, the smaller this 
        # penalty becomes.
        reward = 1 - .1*((abs(current_position - target_position)).sum())**2 
 
        return reward

    def step(self, rotor_speeds):
        
        """Uses action to obtain next state, reward, done."""
        
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            # State size for this task contains only x-y-z coordinates.
            pose_all.append(self.sim.pose[:3]) 
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        
        """Reset the sim to start a new episode."""
        
        self.sim.reset()
        # State size for this task contains only x-y-z coordinates.
        state = np.concatenate([self.sim.pose[:3]] * self.action_repeat) 
        return state