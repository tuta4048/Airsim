"""
For connecting to the AirSim drone environment and testing API functionality
"""
import airsim
import os
import tempfile
import pprint
import numpy as np
from AirSimClient import *


# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)



# state = client.simGetObjectPose("player").position
# pitch, roll, yaw  = client.getPitchRollYaw()
# state = client.getMultirotorState().kinematics_estimated
# print(state)


# AirSimClientBase.wait_key('Press any key to Move by rotor-speed1')
client.moveByRotorSpeed(o0= 900, o1= 899, o2= 900, o3=900, duration= 2) 

# state = client.getMultirotorState().kinematics_estimated
# print(state)

AirSimClientBase.wait_key('Press any key to Move by rotor-speed2')
# client.moveByRotorSpeed(o0= 500, o1= 500, o2= 500, o3=550, duration= 0.001)

def get_current_state():
    input_state = []
    orien = client.getMultirotorState().kinematics_estimated.orientation
    an_vel = client.getMultirotorState().kinematics_estimated.angular_velocity
    lin_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    posit = client.getMultirotorState().kinematics_estimated.position
    input_state = orien.x_val, orien.y_val, orien.z_val, an_vel.x_val, an_vel.y_val, an_vel.z_val, lin_vel.x_val, lin_vel.y_val, lin_vel.z_val, posit.x_val, posit.y_val, posit.z_val 
    print(np.shape(input_state))
    return input_state
    
in_state = get_current_state()
print(in_state)
client.

# AirSimClientBase.wait_key('Press any key to reset')
client.reset() 