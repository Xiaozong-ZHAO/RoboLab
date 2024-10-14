import os 
import numpy as np
from numpy.fft import fft, fftfreq
import time
import random
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
from scipy.signal import hilbert, butter, filtfilt

# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir,use_gui=False)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")



# single joint tuning
#episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, kd, joints_id, regulation_displacement=0.1, episode_duration=20, plot=False):
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # updating the kp value for the joint we want to tune
    kp_vec = np.array([1000]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp
    kd_vec = np.array([0]*dyn_model.getNumberofActuatedJoints())
    kd_vec[joints_id]=kd
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement 

   
    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors


    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    

    steps = int(episode_duration/time_step)
    print("Step Number:", steps)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes[joints_id])
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        print("current time in seconds",current_time, time_step)

    
    # TODO make the plot for the current joint
    # x_axis = len(q_mes_all)
    q_mes_all = np.array(q_mes_all)

    # q_mes_all = q_mes_all - q_des[joint_id]
    # envelop = np.abs(hilbert(q_mes_all))
    # b, a = butter(3, 0.001)
    # smoothed_envelop = filtfilt(b, a, envelop)
    t = np.linspace(0, 20, len(q_mes_all))

    # plt.plot(smoothed_envelop, label='Envelop')
    
    first_derivative = np.diff(q_mes_all) / time_step
    peaks_idx = np.where(np.diff(np.sign(first_derivative)) == -2)[0] + 1
    peak_times = t[peaks_idx]
    peak_values = q_mes_all[peaks_idx]
    filtered_peak_times = peak_times[peak_values > q_des[joint_id]]
    dt = np.diff(filtered_peak_times)
    valid_dt = dt[dt > 0.5]
    filtered_peak_values = peak_values[peak_values > q_des[joint_id]]
    dq = np.diff(filtered_peak_values)
    valid_dq = dq[dt > 0.5]

    slopes = valid_dq / valid_dt
    plt.figure(figsize=(10, 6))
    plt.plot(t, q_mes_all, label='Joint Position')
    print("The time deviations: ", valid_dt)
    print("The differences: ", valid_dq)
    print("The slope values are: ", slopes)
    plt.plot(filtered_peak_times,filtered_peak_values, 'rx', label="Peaks")
    plt.axhline(y=q_des[joints_id], color='orange', linestyle='--', label='y=1')
    plt.title(f"Joint {joints_id+1} Position, Kp={kp}, Kd={kd}")
    plt.show()
    return q_mes_all




def perform_frequency_analysis(data, dt):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])
    power = power/np.max(power[1:])
    dominant_frequency = xf[1:][np.argmax(power[1:])]
    count = count = np.sum(power[1:] > 0.1)
    print("The number of large harmonics: ", count)

    print("frequency sequence is ", xf)
    print("The dominant frequency is: ", dominant_frequency)

    # Optional: Plot the spectrum
    plt.figure()
    plt.plot(xf[1:], power[1:])
    plt.title("FFT of the signal")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    return dominant_frequency, count

def Ku_automation(kd,joints_id, regulation_displacement):
    # Randomly generate a Ku from 0 to 1000
    Ku = random.randint(0, 1000)
    Ku = 30
    # Do simulation
    data = simulate_with_given_pid_values(sim, Ku,kd,joints_id, regulation_displacement,episode_duration=20,plot=False)
    while(True):
        # Conduct frequency analysis
        dominant_frequency, count = perform_frequency_analysis(data, 0.001)
        # Check whether the waveform is chaotic
        if count < 10:
            print("The wavefor is not chaotic!")
        # We analyze the enlope of the waveform
        else:
            print("The waveform is chaotic!")
            # We reduce the Ku value by half
            Ku = Ku - Ku/2

    return Ku


# TODO Implement the table in thi function

if __name__ == '__main__':
    joint_id = 4  # Joint ID to tune
    regulation_displacement = 1  # Displacement from the initial joint position
    init_gain=1000 
    gain_step=1.5 
    max_gain=10000 
    test_duration=20 # in seconds
    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
    kp = 13
    kd = 0
    data=simulate_with_given_pid_values(sim,kp,kd,joint_id,regulation_displacement,episode_duration=20,plot=False)
    # data = Ku_automation(kd,joint_id,regulation_displacement)
    perform_frequency_analysis(data, 0.001)
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method