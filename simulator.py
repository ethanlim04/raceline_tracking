import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from time import time

from racetrack import RaceTrack
from racecar import RaceCar
from controller import lower_controller, controller

class Simulator:
    def __init__(self, rt : RaceTrack):
        matplotlib.rcParams["figure.dpi"] = 250
        matplotlib.rcParams["font.size"] = 8

        self.rt = rt
        self.figure, self.axis = plt.subplots(1, 1)

        self.axis.set_xlabel("X"); self.axis.set_ylabel("Y")

        self.car = RaceCar(self.rt.initial_state.T)

        self.lap_time_elapsed = 0
        self.simulation_time = 0
        self.lap_finished = False
        self.lap_started = False
        
        self.track_limit_violations = 0
        self.currently_violating = False
        self.violation_positions = []
        
        self.total_distance = 0.0
        self.prev_position = self.car.state[0:2].copy()
        
        self.trajectory = []
        # to skip 9 out of 10 frames
        self.steps_per_frame = 10

    def check_track_limits(self):
        car_position = self.car.state[0:2]
        
        min_dist_right = float('inf')
        min_dist_left = float('inf')
        
        for i in range(len(self.rt.right_boundary)):
            dist_right = np.linalg.norm(car_position - self.rt.right_boundary[i])
            dist_left = np.linalg.norm(car_position - self.rt.left_boundary[i])
            
            if dist_right < min_dist_right:
                min_dist_right = dist_right
            if dist_left < min_dist_left:
                min_dist_left = dist_left
        
        centerline_distances = np.linalg.norm(self.rt.centerline - car_position, axis=1)
        closest_idx = np.argmin(centerline_distances)
        
        to_right = self.rt.right_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_left = self.rt.left_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_car = car_position - self.rt.centerline[closest_idx]
        
        right_dist = np.linalg.norm(to_right)
        left_dist = np.linalg.norm(to_left)
        
        proj_right = np.dot(to_car, to_right) / right_dist if right_dist > 0 else 0
        proj_left = np.dot(to_car, to_left) / left_dist if left_dist > 0 else 0
        
        is_violating = proj_right > right_dist or proj_left > left_dist
        
        if is_violating and not self.currently_violating:
            self.track_limit_violations += 1
            self.violation_positions.append(car_position.copy())
            self.currently_violating = True
        elif not is_violating:
            self.currently_violating = False

    def run(self):
        try:
            if self.lap_finished:
                # Stop the timer and show final centered view
                self.timer.stop()
                self.axis.cla()
                self.rt.plot_track(self.axis)
                
                # Center on track
                center_x = np.mean(self.rt.centerline[:, 0])
                center_y = np.mean(self.rt.centerline[:, 1])
                range_x = np.ptp(self.rt.centerline[:, 0])
                range_y = np.ptp(self.rt.centerline[:, 1])
                max_range = max(range_x, range_y) * 0.6
                
                self.axis.set_xlim(center_x - max_range, center_x + max_range)
                self.axis.set_ylim(center_y - max_range, center_y + max_range)

            else:
                self.figure.canvas.flush_events()
                self.axis.cla()
                self.rt.plot_track(self.axis)
                # Zoom out more
                self.axis.set_xlim(self.car.state[0] - 1000, self.car.state[0] + 1000)
                self.axis.set_ylim(self.car.state[1] - 1000, self.car.state[1] + 1000)

            if not self.lap_finished:
                for _ in range(self.steps_per_frame):
                    desired = controller(self.car.state, self.car.parameters, self.rt)
                    cont = lower_controller(self.car.state, desired, self.car.parameters)
                    self.car.update(cont)
                    
                    self.simulation_time += self.car.time_step
                    
                    current_position = self.car.state[0:2]
                    self.total_distance += float(np.linalg.norm(current_position - self.prev_position))
                    self.prev_position = current_position.copy()
                    
                    self.update_status()
                    self.check_track_limits()
                    
                    self.trajectory.append([self.car.state[0], self.car.state[1], self.car.state[3]])

                self.axis.arrow(
                    self.car.state[0], self.car.state[1], \
                    self.car.wheelbase*np.cos(self.car.state[4]), \
                    self.car.wheelbase*np.sin(self.car.state[4])
                )

            # Fixed HUD in top-right corner
            avg_speed = self.total_distance / max(self.lap_time_elapsed, 0.001) if self.lap_started else 0
            
            hud_y = [0.98, 0.95, 0.92, 0.89, 0.86, 0.83]
            self.axis.text(0.98, hud_y[0], f"Lap completed: {self.lap_finished}",
                ha="right", va="top", fontsize=8, color="Red", transform=self.axis.transAxes)
            self.axis.text(0.98, hud_y[1], f"Lap time: {self.lap_time_elapsed:.2f}",
                ha="right", va="top", fontsize=8, color="Red", transform=self.axis.transAxes)
            self.axis.text(0.98, hud_y[2], f"Track violations: {self.track_limit_violations}",
                ha="right", va="top", fontsize=8, color="Red", transform=self.axis.transAxes)
            self.axis.text(0.98, hud_y[3], f"Lap started: {self.lap_started}",
                ha="right", va="top", fontsize=8, color="Red", transform=self.axis.transAxes)
            self.axis.text(0.98, hud_y[4], f"Speed: {self.car.state[3]:.2f} m/s | Avg: {avg_speed:.2f} m/s",
                ha="right", va="top", fontsize=8, color="Red", transform=self.axis.transAxes)
            self.axis.text(0.98, hud_y[5], f"Distance: {self.total_distance:.1f} m",
                ha="right", va="top", fontsize=8, color="Red", transform=self.axis.transAxes)
            
            # Plot trajectory with speed colors
            if len(self.trajectory) > 1:
                trajectory_array = np.array(self.trajectory)
                step = max(1, len(trajectory_array) // 1000)
                for i in range(0, len(trajectory_array) - 1, step):
                    speed = trajectory_array[i, 2]
                    if speed < 20:
                        color = 'red'
                    elif speed < 50:
                        ratio = (speed - 20) / 30
                        color = (1.0, ratio * 0.65, 0.0)
                    else:
                        ratio = (speed - 50) / 50
                        color = ((1.0 - ratio), 0.65 + ratio * 0.35, 0.0)
                    self.axis.plot(trajectory_array[i:i+2, 0], trajectory_array[i:i+2, 1], 
                                 color=color, linewidth=1.5, alpha=0.8)
            
            # Draw red circles at violation positions
            for violation_pos in self.violation_positions:
                self.axis.plot(violation_pos[0], violation_pos[1], 'ro', markersize=6, alpha=0.6)

            self.figure.canvas.draw()
            
            if self.lap_finished:
                plt.show()
                return False
            
            return True

        except KeyboardInterrupt:
            exit()

    def update_status(self):
        progress = np.linalg.norm(self.car.state[0:2] - self.rt.centerline[0, 0:2], 2)

        if progress > 10.0 and not self.lap_started:
            self.lap_started = True
    
        if progress <= 10.0 and self.lap_started and not self.lap_finished:
            self.lap_finished = True
            self.lap_time_elapsed = self.simulation_time
            print(f"Lap completed! Time: {self.lap_time_elapsed:.2f}s, Track violations: {self.track_limit_violations}")

        if not self.lap_finished and self.lap_started:
            self.lap_time_elapsed = self.simulation_time

    def start(self):
        self.timer = self.figure.canvas.new_timer(interval=10)
        self.timer.add_callback(self.run)
        self.timer.start()