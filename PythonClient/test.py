import airsim
import numpy as np
import os
import tempfile
import pprint
import random
from AirSimClient import *


# get input layer value
def get_current_state():
    input_state = []
    orien = client.getMultirotorState().kinematics_estimated.orientation
    an_vel = client.getMultirotorState().kinematics_estimated.angular_velocity
    lin_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    posit = client.getMultirotorState().kinematics_estimated.position
    input_state = orien.x_val, orien.y_val, orien.z_val, an_vel.x_val, an_vel.y_val, an_vel.z_val, lin_vel.x_val, lin_vel.y_val, lin_vel.z_val, posit.x_val, posit.y_val, posit.z_val 
    
    return input_state



#  Connection for Airsim
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


#input variable ******
# input_state = get_current_state()
# print(input_state)
# print([input_state[8]])


def interpret_action(action):           #concering output actions
    scaling_factor = 1000
    
    if action == 0:
        client.moveByRotorSpeed(rp[0], rp[1], rp[2], rp[3], 0.001)
    elif action == 1:
        client.moveByRotorSpeed(rp[0]+scaling_factor, rp[1], rp[2], rp[3], 0.001)
        rp[0] += scaling_factor
    elif action == 2:
        client.moveByRotorSpeed(rp[0], rp[1]+scaling_factor, rp[2], rp[3], 0.001)
        rp[1] += scaling_factor
    elif action == 3:
        client.moveByRotorSpeed(rp[0], rp[1], rp[2]+scaling_factor, rp[3], 0.001)
        rp[2] += scaling_factor
    elif action == 4:
        client.moveByRotorSpeed(rp[0], rp[1], rp[2], rp[3]+scaling_factor, 0.001)
        rp[3] += scaling_factor
    return rp


rp=[0, 0, 0, 0]
interpret_action(random.randint(0,4))
AirSimClientBase.wait_key('Press any key')

interpret_action(random.randint(0,4))
AirSimClientBase.wait_key('Press any key')

interpret_action(random.randint(0,4))
AirSimClientBase.wait_key('Press any key')
interpret_action(random.randint(0,4))
AirSimClientBase.wait_key('Press any key')
interpret_action(random.randint(0,4))
AirSimClientBase.wait_key('Press any key')




AirSimClientBase.wait_key('Press any key to reset')
client.reset() 