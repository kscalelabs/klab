"""Generate a trajectory for the Feetech servos."""
import argparse
import csv
import json
import math
import numpy as np
import random
import time
from datetime import datetime
from typing import List

import pykos


def cubic_interpolate(keyframes: list, t: float):
    if t < keyframes[0][0]:
        return keyframes[0][1]
    if t > keyframes[-1][0]:
        return keyframes[-1][1]

    for i in range(len(keyframes) - 1):
        if keyframes[i][0] <= t <= keyframes[i + 1][0]:
            t0, x0, x0p = keyframes[i]
            t1, x1, x1p = keyframes[i + 1]

            A = [
                [1, t0, t0**2, t0**3],
                [0, 1, 2 * t0, 3 * t0**2],
                [1, t1, t1**2, t1**3],
                [0, 1, 2 * t1, 3 * t1**2],
            ]
            b = [x0, x0p, x1, x1p]
            w = np.linalg.solve(A, b)

            return w[0] + w[1] * t + w[2] * t**2 + w[3] * t**3


class Trajectory:
    duration = None

    def __call__(self, t: float):
        """
        Retrieve (angle, torque_enable) at time t
        """
        raise NotImplementedError


class LiftAndDrop(Trajectory):
    duration = 6.0

    def __init__(self):
        self.drop_time = random.uniform(1.5, 2.5)  # Random drop timing
        self.drop_angle = random.uniform(-np.pi/3, -2*np.pi/3)  # Random drop angle

    def __call__(self, t: float):
        keyframes = [[0.0, 0.0, 0.0], [self.drop_time, self.drop_angle, 0.0]]
        angle = cubic_interpolate(keyframes, t)
        enable = t < self.drop_time
        return angle, enable


class SinusTimeSquare(Trajectory):
    duration = 6.0

    def __init__(self):
        self.frequency = random.uniform(0.5, 2.0)  # Random frequency multiplier

    def __call__(self, t: float):
        angle = np.sin((t * self.frequency)**2)
        return angle, True


class UpAndDown(Trajectory):
    duration = 6.0

    def __init__(self):
        self.peak_time = random.uniform(2.5, 3.5)  # Random peak timing
        self.peak_angle = random.uniform(np.pi/3, 2*np.pi/3)  # Random peak angle
        self.final_ratio = random.uniform(0.7, 0.9)  # Random final position ratio

    def __call__(self, t: float):
        keyframes = [
            [0.0, 0.0, 0.0],
            [self.peak_time, self.peak_angle, 0.0],
            [6.0, self.final_ratio * self.peak_angle, 0.0],
        ]
        angle = cubic_interpolate(keyframes, t)
        return angle, True


class SinSin(Trajectory):
    duration = 6.0

    def __init__(self):
        self.primary_freq = random.uniform(0.8, 1.2)  # Base frequency variation
        self.secondary_freq = random.uniform(4.0, 6.0)  # Higher frequency variation
        self.modulation_freq = random.uniform(1.5, 2.5)  # Modulation frequency
        self.secondary_amp = random.uniform(0.3, 0.7)  # Secondary amplitude

    def __call__(self, t: float):
        angle = (np.sin(t * self.primary_freq) * np.pi / 2 + 
                np.sin(self.secondary_freq * t) * 
                self.secondary_amp * 
                np.sin(t * self.modulation_freq))
        return angle, True


class Nothing(Trajectory):
    duration = 6.0

    def __init__(self):
        self.duration = random.uniform(1.0, 4.0)  # Random duration of doing nothing

    def __call__(self, t: float):
        return 0.0, False


# Update trajectories dictionary to create new instances when selected
def get_random_trajectory(name):
    trajectory_classes = {
        "lift_and_drop": LiftAndDrop,
        "sin_time_square": SinusTimeSquare,
        "up_and_down": UpAndDown,
        "sin_sin": SinSin,
        "nothing": Nothing,
    }
    return trajectory_classes[name]()


trajectories = {
    "lift_and_drop": LiftAndDrop(),
    "sin_time_square": SinusTimeSquare(),
    "up_and_down": UpAndDown(),
    "sin_sin": SinSin(),
    "nothing": Nothing(),
}

# IP configuration
usb = "192.168.42.1"
pawel = "10.33.12.222"
rui = "192.168.2.206"

ip_aliases = {
    "pawel": pawel,
    "rui": rui,
    "usb": usb
}

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-actuator sine wave test')
    parser.add_argument('--ipalias', type=str, default='pawel', help='IP alias of the KOS')
    parser.add_argument('--ids', type=int, nargs='+', default=[1, 2, 3], help='List of actuator IDs')
    parser.add_argument('--frequency', type=float, default=3.0, help='Sine wave frequency in Hz')
    parser.add_argument('--amplitude', type=float, default=30.0, help='Amplitude in degrees')
    parser.add_argument('--duration', type=float, default=50.0, help='Test duration in seconds')
    parser.add_argument('--phase-offset', type=float, default=15.0, 
                        help='Phase offset between successive actuators in degrees')
    parser.add_argument('--kp', type=float, default=64.0, help='Position gain')
    parser.add_argument('--kd', type=float, default=64.0, help='Velocity gain')
    parser.add_argument('--log', action='store_true', help='Log the results to a file')
    parser.add_argument('--control-freq', type=float, default=50.0, help='Control frequency in Hz')
    parser.add_argument('--random-runs', type=int, default=0, 
                       help='Number of random configuration runs to perform')
    parser.add_argument('--zero-offset', action='store_true', 
                       help='Use zero as initial position instead of current position')
    return parser.parse_args()

def configure_actuators(actuator_controller, ids: List[int], kp: float, kd: float):
    """Configure all actuators with the specified gains"""
    for id in ids:
        actuator_controller.configure_actuator(
            actuator_id=id,
            kp=kp,
            kd=kd,
            torque_enabled=True
        )

def setup_logging(args, initial_positions):
    """Setup logging files and save configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'logs/sine_test_{timestamp}'
    
    # Save configuration to JSON
    config = {
        'timestamp': timestamp,
        'actuator_ids': args.ids,
        'frequency': args.frequency,
        'amplitude': args.amplitude,
        'phase_offset': args.phase_offset,
        'duration': args.duration,
        'kp': args.kp,
        'kd': args.kd,
        'control_freq': args.control_freq,
        'initial_positions': initial_positions,
        'data': []  # Will store all data points
    }
    
    # Instead of opening CSV file for writing, just store rows in memory
    if args.log:
        csv_rows = []  # Store rows in memory
        header = ['timestamp', 'timestep']
        for id in args.ids:
            header.extend([
                f'actuator_{id}_initial_offset',
                f'actuator_{id}_current_state',
                f'actuator_{id}_commanded_state',
                f'actuator_{id}_relative_position'
            ])
        csv_rows.append(header)  # Add header as first row
    
    return csv_rows, config, base_filename

def run_sine_test(actuator_controller, 
                  args: argparse.Namespace,
                  ids: List[int],
                  frequency: float,
                  amplitude: float,
                  duration: float,
                  phase_offset: float,
                  control_freq: float,
                  initial_positions: dict = None):
    """Run wave test on multiple actuators with phase offsets"""
    
    # Expand wave types to include trajectories
    # wave_types = ['sine', 'square', 'triangle']
    wave_types = ["square"]
    trajectory_types = list(trajectories.keys())

    all_patterns = wave_types + trajectory_types
    
    current_pattern = random.choice(all_patterns)
    current_trajectory = None
    if current_pattern in trajectory_types:
        current_trajectory = trajectories[current_pattern]
        pattern_duration = current_trajectory.duration
    else:
        periods_to_run = random.randint(1, 4)
        pattern_duration = periods_to_run / frequency
    
    next_pattern_change = time.time() + pattern_duration
    pattern_start_time = time.time()
    
    # Get initial positions based on configuration
    if args.zero_offset:
        initial_positions = {id: 0.0 for id in ids}
    elif initial_positions is None:
        initial_positions = {}
        states = actuator_controller.get_actuators_state(ids)
        for state in states:
            initial_positions[state.actuator_id] = state.position
    
    # Setup logging if enabled
    if args.log:
        csv_rows, config, base_filename = setup_logging(args, initial_positions)
    
    start_time = time.time()
    timestep = 0
    
    # Add timing info to config if logging
    if args.log:
        config['timing_data'] = []
    
    # Pre-allocate actuator commands list to avoid recreating it each loop
    actuator_commands = [{"actuator_id": id, "position": 0.0} for id in ids]

    actuator_controller.command_actuators(actuator_commands)
    time.sleep(2)
    
    try:
        while time.time() - start_time < duration:
            process_start = time.time()
            
            # Check if it's time to change the pattern
            if time.time() >= next_pattern_change:
                current_pattern = random.choice(all_patterns)
                pattern_start_time = time.time()
                
                if current_pattern in trajectory_types:
                    current_trajectory = get_random_trajectory(current_pattern)
                    pattern_duration = current_trajectory.duration
                    print(f"Switching to trajectory: {current_pattern} for {pattern_duration:.1f} seconds")
                else:
                    current_trajectory = None
                    periods_to_run = random.randint(1, 3)
                    pattern_duration = periods_to_run / frequency
                    print(f"Switching to {current_pattern} wave for {periods_to_run} periods")
                
                next_pattern_change = time.time() + pattern_duration
            
            cur_states = actuator_controller.get_actuators_state(ids)
            state_read_time = time.time()
            
            t = timestep / control_freq
            pattern_time = time.time() - pattern_start_time
            
            # Update commands based on current pattern
            for idx, command in enumerate(actuator_commands):
                # pfb30
                if True:
                    # Use standard wave patterns
                    phase = 2 * math.pi * frequency * t + math.radians(phase_offset) * idx
                    wave_value = 1.0 if math.sin(phase) >= 0 else -1.0

                    command["position"] = initial_positions[command["actuator_id"]] + amplitude * wave_value
                elif current_trajectory is not None:
                    # Use trajectory
                    angle, enable = current_trajectory(pattern_time)
                    wave_value = angle / (np.pi/2)  # Normalize to [-1, 1] range
                    if not enable:
                        command["position"] = initial_positions[command["actuator_id"]]
                    else:
                        command["position"] = initial_positions[command["actuator_id"]] + amplitude * wave_value
                else:
                    # Use standard wave patterns
                    phase = 2 * math.pi * frequency * t + math.radians(phase_offset) * idx
                    
                    if current_pattern == 'sine':
                        wave_value = math.sin(phase)
                    elif current_pattern == 'square':
                        breakpoint()
                        wave_value = 1.0 if math.sin(phase) >= 0 else -1.0
                    else:  # triangle
                        wave_value = 2 * abs((phase / math.pi) % 2 - 1) - 1
                    
                    command["position"] = initial_positions[command["actuator_id"]] + amplitude * wave_value
            
            command_prep_time = time.time()
            
            # Send all commands in one batch
            actuator_controller.command_actuators(actuator_commands)
            command_send_time = time.time()
            
            # Calculate timing information
            timing_info = {
                'timestep': timestep,
                'total_loop_start': process_start - start_time,
                'state_read_duration': state_read_time - process_start,
                'command_prep_duration': command_prep_time - state_read_time,
                'command_send_duration': command_send_time - command_prep_time,
                'total_process_duration': command_send_time - process_start
            }
            
            if args.log:
                config['timing_data'].append(timing_info)
                
                # Prepare logging row
                timestamp = datetime.now().isoformat()
                log_row = [timestamp, timestep]
                data_point = {
                    'timestamp': timestamp,
                    'timestep': timestep,
                    'actuators': {}
                }
                
                # Log data for each actuator
                for id in ids:
                    current_state = next((state.position for state in cur_states if state.actuator_id == id), None)
                    commanded_position = next((cmd["position"] for cmd in actuator_commands if cmd["actuator_id"] == id), None)
                    relative_position = current_state - initial_positions[id] if current_state is not None else None
                    
                    # Add to CSV row
                    log_row.extend([
                        initial_positions[id],
                        current_state,
                        commanded_position,
                        relative_position
                    ])
                    
                    # Add to JSON data point
                    data_point['actuators'][id] = {
                        'initial_offset': initial_positions[id],
                        'current_state': current_state,
                        'commanded_state': commanded_position,
                        'relative_position': relative_position
                    }
                
                csv_rows.append(log_row)
                config['data'].append(data_point)
            
            process_end = time.time()
            sleep_duration = max(0, 1 / control_freq - (process_end - process_start))
            time.sleep(sleep_duration)
            
            if args.log:
                timing_info['sleep_duration'] = sleep_duration
                timing_info['actual_loop_duration'] = time.time() - process_start
            
            timestep += 1
    
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Stopping test.")
    finally:
        if args.log:
            # Write all CSV data at once
            with open(f'{base_filename}_data.csv', 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(csv_rows)
            
            # Add timing statistics and save JSON
            if config['timing_data']:
                timing_stats = {
                    'mean_process_duration': sum(t['total_process_duration'] for t in config['timing_data']) / len(config['timing_data']),
                    'max_process_duration': max(t['total_process_duration'] for t in config['timing_data']),
                    'min_process_duration': min(t['total_process_duration'] for t in config['timing_data']),
                    'mean_loop_duration': sum(t['actual_loop_duration'] for t in config['timing_data']) / len(config['timing_data']),
                    'achieved_frequency': len(config['timing_data']) / (config['timing_data'][-1]['total_loop_start'])
                }
                config['timing_stats'] = timing_stats
            
            # Save complete dataset to JSON
            with open(f'{base_filename}_data.json', 'w') as f:
                json.dump(config, f, indent=4)
        
        # Disable all actuators
        for id in ids:
            actuator_controller.configure_actuator(actuator_id=id, kp=32, kd=32, torque_enabled=False)


def run_random_tests(actuator_controller, 
                    base_args: argparse.Namespace,
                    num_runs: int,
                    freq_range=(0.1, 2.0),
                    amp_range=(10.0, 30.0),
                    phase_range=(0.0, 30.0),
                    kp_range=(32.0, 64.0),
                    kd_range=(32.0, 64.0),
                    control_freq_range=(25.0, 100.0)):
    """
    Run multiple sine tests with randomized parameters
    
    Args:
        actuator_controller: KOS actuator controller
        base_args: Base configuration from argparse
        num_runs: Number of random tests to run
        freq_range: (min, max) frequency in Hz
        amp_range: (min, max) amplitude in degrees
        phase_range: (min, max) phase offset in degrees
        kp_range: (min, max) position gain
        kd_range: (min, max) velocity gain
        control_freq_range: (min, max) control frequency in Hz
    """
    # Get initial positions if not using zero offset
    initial_positions = None
    if not base_args.zero_offset:
        initial_positions = {}
        states = actuator_controller.get_actuators_state(base_args.ids)
        for state in states:
            initial_positions[state.actuator_id] = state.position

    try:
        for run in range(num_runs):
            # Create a copy of the base arguments dictionary
            random_args_dict = vars(base_args).copy()
            
            # Update with random values
            random_args_dict.update({
                'frequency': random.uniform(*freq_range),
                'amplitude': random.uniform(*amp_range),
                'phase_offset': random.uniform(*phase_range),
                # 'kp': random.uniform(*kp_range),
                # 'kd': random.uniform(*kd_range),
                # 'control_freq': random.uniform(*control_freq_range)
            })
            
            # Create new Namespace from updated dictionary
            random_args = argparse.Namespace(**random_args_dict)
            
            print(f"\nStarting run {run + 1}/{num_runs} with parameters:")
            print(f"Frequency: {random_args.frequency:.2f} Hz")
            print(f"Amplitude: {random_args.amplitude:.2f} degrees")
            print(f"Phase offset: {random_args.phase_offset:.2f} degrees")
            print(f"Kp: {random_args.kp:.2f}")
            print(f"Kd: {random_args.kd:.2f}")
            print(f"Control Frequency: {random_args.control_freq:.2f} Hz")
            
            # Configure actuators with new gains
            configure_actuators(actuator_controller, random_args.ids, 
                              random_args.kp, random_args.kd)
            
            # Run the test with random parameters
            run_sine_test(
                actuator_controller,
                random_args,
                random_args.ids,
                random_args.frequency,
                random_args.amplitude,
                random_args.duration,
                random_args.phase_offset,
                random_args.control_freq,
                initial_positions=initial_positions
            )
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping random tests.")
    finally:
        # Disable all actuators
        for id in base_args.ids:
            actuator_controller.configure_actuator(
                actuator_id=id, 
                kp=32, 
                kd=32, 
                torque_enabled=False
            )

def main():
    args = parse_args()
    
    # Connect to KOS
    kos = pykos.KOS(ip=ip_aliases[args.ipalias])
    ac = kos.actuator
    
    # Configure all actuators
    configure_actuators(ac, args.ids, args.kp, args.kd)

    if args.random_runs > 0:
        print(f"Running {args.random_runs} random configuration tests")
        run_random_tests(ac, args, args.random_runs)
    else:
        # Original single test code
        print("Running sine test with parameters:")
        print(f"Actuator IDs: {args.ids}")
        print(f"Frequency: {args.frequency} Hz")
        print(f"Amplitude: {args.amplitude} degrees")
        print(f"Phase offset: {args.phase_offset} degrees")
        print(f"Duration: {args.duration} seconds")
        
        run_sine_test(
            ac,
            args,
            args.ids,
            args.frequency,
            args.amplitude,
            args.duration,
            args.phase_offset,
            args.control_freq
        )

    for id in args.ids:
        ac.configure_actuator(actuator_id=id, kp=32, kd=32, torque_enabled=False)

if __name__ == "__main__":
    main()
