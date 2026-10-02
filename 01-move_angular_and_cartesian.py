#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading

# Add the examples folder to sys.path so we can import utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.join(script_dir, "..")  # utilities.py is in the examples folder
sys.path.insert(0, examples_dir)

import utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# Closure to set an event after END or ABORT
def check_for_end_or_abort(e):
    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
            e.set()
    return check

def example_move_to_home_position(base):
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = next((a.handle for a in action_list.action_list if a.name == "Home"), None)

    if action_handle is None:
        print("Can't reach safe position. Exiting")
        return False
=
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    print("Safe position reached" if finished else "Timeout on action notification wait")
    return finished

def example_angular_action_movement(base):
    print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = 0

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())

    print("Executing action")
    base.ExecuteAction(action)

    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    print("Angular movement completed" if finished else "Timeout on action notification wait")
    return finished

def example_cartesian_action_movement(base, base_cyclic):
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()
    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x
    cartesian_pose.y = feedback.base.tool_pose_y - 0.1
    cartesian_pose.z = feedback.base.tool_pose_z - 0.2
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())

    print("Executing action")
    base.ExecuteAction(action)

    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    print("Cartesian movement completed" if finished else "Timeout on action notification wait")
    return finished

def main():
    # Parse arguments
    args = utilities.parseConnectionArguments()

    # Connect to the device
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        success = True
        success &= example_move_to_home_position(base)
        success &= example_cartesian_action_movement(base, base_cyclic)
        success &= example_angular_action_movement(base)

        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
