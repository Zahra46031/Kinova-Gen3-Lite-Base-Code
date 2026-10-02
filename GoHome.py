#!/usr/bin/env python3
import sys
import os
import time
import threading
import numpy as np

# ---- Path fix for utilities.py ----
script_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.join(script_dir, "..")
sys.path.insert(0, examples_dir)

import utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

TIMEOUT_DURATION = 20
HOME_K = np.array([0.3, 0.06, 0.2])  # meters, Kinova frame

# ============================================================
# FRAME TRANSFORMS
# ============================================================

def kinova_to_mujoco(p_k):
    """
    Kinova base frame → MuJoCo world frame
    X_k (up)       → Z_mj
    Y_k (left)     → Y_mj
    Z_k (forward)  → X_mj
    """
    return np.array([p_k[2], p_k[1], p_k[0]], dtype=np.float32)


def mujoco_to_kinova(p_mj):
    """
    MuJoCo world frame → Kinova base frame
    """
    return np.array([p_mj[2], p_mj[1], p_mj[0]], dtype=np.float32)


# ============================================================
# ACTION END HANDLER
# ============================================================

def check_for_end_or_abort(e):
    def check(notification, e=e):
        if notification.action_event in (
            Base_pb2.ACTION_END,
            Base_pb2.ACTION_ABORT,
        ):
            e.set()
    return check


# ============================================================
# GET CURRENT CARTESIAN POSE
# ============================================================

def get_current_pose(base_cyclic):
    fb = base_cyclic.RefreshFeedback()
    pos_k = np.array([
        fb.base.tool_pose_x,
        fb.base.tool_pose_y,
        fb.base.tool_pose_z
    ])
    ori_k = np.array([
        fb.base.tool_pose_theta_x,
        fb.base.tool_pose_theta_y,
        fb.base.tool_pose_theta_z
    ])
    return pos_k, ori_k


# ============================================================
# MOVE CARTESIAN
# ============================================================

def move_cartesian(base, base_cyclic, target_k):
    action = Base_pb2.Action()
    action.name = "CartesianMove"

    fb = base_cyclic.RefreshFeedback()
    pose = action.reach_pose.target_pose

    pose.x, pose.y, pose.z = target_k
    pose.theta_x = fb.base.tool_pose_theta_x
    pose.theta_y = fb.base.tool_pose_theta_y
    pose.theta_z = fb.base.tool_pose_theta_z

    e = threading.Event()
    h = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteAction(action)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(h)

    return finished


# ============================================================
# MOVE TO HOME
# ============================================================

def move_to_home(base):
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)

    home = next((a.handle for a in action_list.action_list if a.name == "Home"), None)
    if home is None:
        print("❌ Home action not found")
        return False

    e = threading.Event()
    h = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(home)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(h)

    return finished


# ============================================================
# MAIN
# ============================================================

def main():
    args = utilities.parseConnectionArguments()

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        print("✅ Connected to Kinova")

        # ----------------------------------------------------
        # Move to Home
        # ----------------------------------------------------
        print("\n➡ Moving to HOME")
        #move_to_home(base)
        print("\n➡ Moving to SOFTWARE HOME")
        move_cartesian(base, base_cyclic, HOME_K)

        time.sleep(1)

        # Read home pose
        home_k, _ = get_current_pose(base_cyclic)
        home_mj = kinova_to_mujoco(home_k)

        print("\n🏠 HOME POSITION")
        print("Kinova:", home_k)
        print("MuJoCo:", home_mj)

        # ----------------------------------------------------
        # Define TARGET IN MUJOCO FRAME
        # ----------------------------------------------------
        target_mj = np.array([
            home_mj[0] + 0.30,  # forward
            home_mj[1] + 0.10,  # left
            home_mj[2] - 0.20   # down
        ])

        target_k = mujoco_to_kinova(target_mj)

        print("\n🎯 TARGET")
        print("MuJoCo:", target_mj)
        print("Kinova:", target_k)

        # ----------------------------------------------------
        # Move to target
        # ----------------------------------------------------
        print("\n➡ Moving to TARGET")
        move_cartesian(base, base_cyclic, target_k)
        time.sleep(1)

        # ----------------------------------------------------
        # Return Home
        # ----------------------------------------------------
        print("\n↩ Returning to HOME")
        #move_to_home(HOME_K)
        move_cartesian(base, base_cyclic, HOME_K)

        print("\n✅ Done")


if __name__ == "__main__":
    main()