"""HTTP client for robot and hand control through franka_server.py"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R


class HTTPRobotClient:
    """HTTP client for robot arm control"""

    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: float = 1.0):
        """
        Initialize HTTP robot client.
        
        Args:
            base_url: Base URL of the franka_server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.state_history = []
        self.max_history_size = 1000
        
    def schedule_waypoint(self, target_pose: np.ndarray, target_time: float) -> bool:
        """
        Send pose command to robot.
        
        Args:
            target_pose: 6D pose (xyz + rotation vector) or 7D (xyz + quaternion)
            target_time: Target time for execution (not used in HTTP mode)
        
        Returns:
            Success status
        """
        try:
            # Convert 6D pose to 7D if needed
            if len(target_pose) == 6:
                xyz = target_pose[:3]
                quat = R.from_rotvec(target_pose[3:]).as_quat()
                pose_7d = np.concatenate([xyz, quat])
            else:
                pose_7d = target_pose
                
            response = requests.post(
                f"{self.base_url}/pose",
                json={"arr": pose_7d.tolist()},
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending waypoint: {e}")
            return False
    
    def get_state(self) -> Dict:
        """
        Get current robot state.
        
        Returns:
            State dict with 'state' and 'receive_time' keys
        """
        try:
            response = requests.post(
                f"{self.base_url}/getstate",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                # Convert to expected format
                state = {
                    "state": {
                        "ActualTCPPose": data["pose"],  # xyz + quat
                        "TargetTCPPose": data["pose"],
                        "ActualQ": data["q"],
                        "ActualQd": data["dq"],
                    },
                    "receive_time": time.time()
                }
                # Store in history
                self.state_history.append(state)
                if len(self.state_history) > self.max_history_size:
                    self.state_history.pop(0)
                return state
        except Exception as e:
            print(f"Error getting state: {e}")
            return {
                "state": {
                    "ActualTCPPose": [0] * 7,
                    "TargetTCPPose": [0] * 7,
                    "ActualQ": [0] * 7,
                    "ActualQd": [0] * 7,
                },
                "receive_time": time.time()
            }
    
    def get_state_history(self) -> List[Dict]:
        """Get state history for interpolation."""
        return self.state_history
    
    def receive_data(self) -> List[Dict]:
        """Get all accumulated state data (compatibility method)."""
        history = self.state_history.copy()
        self.state_history.clear()
        return history


class HTTPHandClient:
    """HTTP client for dexterous hand control"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: float = 1.0):
        """
        Initialize HTTP hand client.
        
        Args:
            base_url: Base URL of the franka_server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        
    def schedule_waypoint(self, target_pos: np.ndarray, target_time: float) -> bool:
        """
        Send hand pose command.
        
        Args:
            target_pos: Target joint positions
            target_time: Target time for execution (not used in HTTP mode)
        
        Returns:
            Success status
        """
        try:
            response = requests.post(
                f"{self.base_url}/hand_pose",
                json={"arr": target_pos.tolist()},
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending hand pose: {e}")
            return False
    
    def get_pos(self) -> np.ndarray:
        """
        Get current hand joint positions.
        
        Returns:
            Current joint positions
        """
        try:
            response = requests.post(
                f"{self.base_url}/get_handangle",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return np.array(response.json()["hand_angle"])
        except Exception as e:
            print(f"Error getting hand position: {e}")
        return np.zeros(12)  # Default for XHand
    
    def get_tactile(self, calc: bool = True) -> np.ndarray:
        """
        Get tactile sensor data.
        
        Args:
            calc: Whether to calculate (compatibility parameter)
        
        Returns:
            Tactile data array
        """
        try:
            response = requests.post(
                f"{self.base_url}/get_handtactile",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return np.array(response.json()["tactile_data"])
        except Exception as e:
            print(f"Error getting tactile data: {e}")
        return np.zeros((5, 3))  # Default 5 fingers, 3 values each
    
    def get_dof(self) -> int:
        """
        Get hand degrees of freedom.
        
        Returns:
            Number of DOFs
        """
        try:
            response = requests.post(
                f"{self.base_url}/get_handdof",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()["dof_num"]
        except Exception as e:
            print(f"Error getting DOF: {e}")
        return 12  # Default for XHand