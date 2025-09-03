import time
import traceback
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from dexumi.camera.camera import Camera, FrameData


def get_all_realsense_cameras():
    """
    Get all connected RealSense cameras.
    """
    ctx = rs.context()
    devices_id = []
    for device in ctx.devices:
        print(
            f"Found device: {device.get_info(rs.camera_info.name)} "
            f"(Serial: {device.get_info(rs.camera_info.serial_number)})"
        )
        devices_id.append(device.get_info(rs.camera_info.serial_number))
    # sort devices_id
    devices_id.sort()
    return devices_id


class RealSenseCamera(Camera):
    def __init__(
        self,
        camera_name: str,
        camera_resolution: Tuple[int, int] = (640, 480),
        latency: Optional[float] = 0.1,
        fps: int = 30,
        device_id: str = None,
        enable_depth: bool = True,
        align_to_color: bool = True,
    ):
        super().__init__(
            camera_name=camera_name,
            camera_resolution=camera_resolution,
            latency=latency,
        )

        self.device_id = device_id
        self.fps = fps
        self.enable_depth = enable_depth
        self.align_to_color = align_to_color

        # Create pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure device if specified
        if self.device_id:
            self.config.enable_device(self.device_id)

        # Configure streams
        self.config.enable_stream(
            rs.stream.color,
            camera_resolution[0],
            camera_resolution[1],
            rs.format.bgr8,
            fps,
        )

        if enable_depth:
            self.config.enable_stream(
                rs.stream.depth,
                camera_resolution[0],
                camera_resolution[1],
                rs.format.z16,
                fps,
            )

        self.is_running = False

        if align_to_color:
            self.align = rs.align(rs.stream.color)

    def get_camera_frame(self):
        if self.is_running:
            try:
                # Wait for a coherent pair of frames
                frames = self.pipeline.wait_for_frames()
                receive_time = time.monotonic()

                if self.align_to_color and self.enable_depth:
                    frames = self.align.process(frames)

                # Get color frame
                color_frame = frames.get_color_frame()
                if not color_frame:
                    return None

                # Convert to numpy array and make deep copy
                rgb = np.copy(np.asanyarray(color_frame.get_data()))

                # Get depth frame if enabled
                depth = None
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth = np.copy(np.asanyarray(depth_frame.get_data()))

                if self.latency is not None:
                    capture_time = receive_time - self.latency
                else:
                    capture_time = receive_time

                return FrameData(
                    rgb=rgb,
                    depth=depth,
                    capture_time=capture_time,
                    receive_time=receive_time,
                )

            except Exception as e:
                traceback.print_exc()
                print("Error: ", e)
                return None

    def start_streaming(self):
        try:
            # Get device info and validate color stream support
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = self.config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()

            found_color = False
            for sensor in device.sensors:
                # Check if sensor supports color stream
                try:
                    profiles = sensor.get_stream_profiles()
                    for profile in profiles:
                        if profile.stream_type() == rs.stream.color:
                            found_color = True
                            break
                    if found_color:
                        break
                except:
                    continue

            if not found_color:
                raise RuntimeError("The RealSense device does not support color stream")

            # Start streaming
            self.profile = self.pipeline.start(self.config)

            # Get depth sensor properties if enabled
            if self.enable_depth:
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                print(f"Depth Scale: {self.depth_scale}")

            self.is_running = True
            print(f"Camera-{self.camera_name} started.")
            print(f"Device name: {device.get_info(rs.camera_info.name)}")
            print(f"Serial number: {device.get_info(rs.camera_info.serial_number)}")
            print(
                f"Firmware version: {device.get_info(rs.camera_info.firmware_version)}"
            )

        except Exception as e:
            traceback.print_exc()
            print("Failed to start streaming:", e)

    def stop_streaming(self):
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
            print(f"Camera-{self.camera_name} stopped.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    all_cam = get_all_realsense_cameras()
    if all_cam:
        cam = RealSenseCamera(
            camera_name="RealSense Camera",
            device_id=all_cam[0],
            enable_depth=True,
            align_to_color=True,
        )
        cam.start_streaming()
        cam.live_streaming()
        cam.stop_streaming()
    else:
        print("No RealSense cameras found")
