#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import time

class RealSenseVisualizer:
    def __init__(self):
        self.context = rs.context()
        self.pipelines = []
        self.configs = []
        self.devices_info = []
        
        self.setup_cameras()
    
    def setup_cameras(self):
        devices = self.context.query_devices()
        print(f"Found {len(devices)} RealSense devices:")
        
        for i, device in enumerate(devices):
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            print(f"  Device {i}: {name} (Serial: {serial})")
            
            # Create pipeline and config for each device
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable device by serial number
            config.enable_device(serial)
            
            # Configure streams based on camera model
            if "L515" in name:
                # L515 specific configuration - it only supports 320x240 for depth
                print(f"    Configuring L515 with native resolutions")
                config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            elif "D405" in name or "D435" in name or "D455" in name:
                # D400 series configuration
                print(f"    Configuring D-series with standard resolutions")
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            else:
                # Default configuration
                print(f"    Using default configuration")
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            try:
                pipeline.start(config)
                self.pipelines.append(pipeline)
                self.configs.append(config)
                self.devices_info.append({'name': name, 'serial': serial, 'index': i})
                print(f"    Successfully initialized {name}")
            except Exception as e:
                print(f"    Failed to initialize {name}: {e}")
    
    def visualize(self):
        if not self.pipelines:
            print("No cameras initialized!")
            return
        
        print("\nStarting visualization...")
        print("Press 'q' to quit, 's' to save images, 'c' to toggle colormap")
        
        colormap = cv2.COLORMAP_JET
        colormap_index = 0
        colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_COOL, cv2.COLORMAP_VIRIDIS]
        colormap_names = ['JET', 'HOT', 'COOL', 'VIRIDIS']
        
        try:
            while True:
                for i, (pipeline, device_info) in enumerate(zip(self.pipelines, self.devices_info)):
                    try:
                        # Get frames
                        frames = pipeline.wait_for_frames()
                        depth_frame = frames.get_depth_frame()
                        color_frame = frames.get_color_frame()
                        
                        if not depth_frame or not color_frame:
                            continue
                        
                        # Convert to numpy arrays
                        depth_image = np.asanyarray(depth_frame.get_data())
                        color_image = np.asanyarray(color_frame.get_data())
                        
                        # Apply colormap to depth image
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_image, alpha=0.03), 
                            colormap
                        )
                        
                        # Create side-by-side display
                        display_height = 480
                        display_width = 640
                        
                        # Resize images if necessary
                        color_resized = cv2.resize(color_image, (display_width, display_height))
                        depth_resized = cv2.resize(depth_colormap, (display_width, display_height))
                        
                        # Add resolution info
                        depth_res = f"{depth_image.shape[1]}x{depth_image.shape[0]}"
                        color_res = f"{color_image.shape[1]}x{color_image.shape[0]}"
                        
                        # Add text labels
                        cv2.putText(color_resized, f"{device_info['name']} - RGB ({color_res})", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(depth_resized, f"{device_info['name']} - Depth ({depth_res}) {colormap_names[colormap_index]}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Add depth statistics
                        depth_stats = f"Min: {np.min(depth_image)}mm, Max: {np.max(depth_image)}mm, Mean: {np.mean(depth_image):.1f}mm"
                        cv2.putText(depth_resized, depth_stats, 
                                  (10, display_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Combine images horizontally
                        combined = np.hstack([color_resized, depth_resized])
                        
                        # Display
                        window_name = f"RealSense {device_info['name']} (Serial: {device_info['serial']})"
                        cv2.imshow(window_name, combined)
                        
                    except Exception as e:
                        print(f"Error processing frames from {device_info['name']}: {e}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current images
                    timestamp = int(time.time())
                    for i, (pipeline, device_info) in enumerate(zip(self.pipelines, self.devices_info)):
                        try:
                            frames = pipeline.wait_for_frames()
                            depth_frame = frames.get_depth_frame()
                            color_frame = frames.get_color_frame()
                            
                            if depth_frame and color_frame:
                                depth_image = np.asanyarray(depth_frame.get_data())
                                color_image = np.asanyarray(color_frame.get_data())
                                
                                # Save images
                                cv2.imwrite(f"realsense_{device_info['serial']}_color_{timestamp}.png", color_image)
                                cv2.imwrite(f"realsense_{device_info['serial']}_depth_{timestamp}.png", depth_image)
                                print(f"Saved images for {device_info['name']}")
                        except Exception as e:
                            print(f"Error saving images for {device_info['name']}: {e}")
                elif key == ord('c'):
                    # Toggle colormap
                    colormap_index = (colormap_index + 1) % len(colormaps)
                    colormap = colormaps[colormap_index]
                    print(f"Changed colormap to: {colormap_names[colormap_index]}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("\nCleaning up...")
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except Exception as e:
                print(f"Error stopping pipeline: {e}")
        cv2.destroyAllWindows()

def main():
    visualizer = RealSenseVisualizer()
    visualizer.visualize()

if __name__ == "__main__":
    main()