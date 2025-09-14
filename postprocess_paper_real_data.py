import numpy as np
# from pymlg import SE3, SO3, SO2
from utils import readAgisoftTrajectory, correct_timestamps
from pathlib import Path
from uvnav_py.plotter import plot_poses
from uvnav_py.states import SE3State
from uvnav_py.bspline import SE3Bspline
from uvnav_py.utils import state_interp

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
from pymlg.numpy import SE3, SO3, SE2, SO2
from datetime import datetime
import pandas as pd


## Compute angular velocities from rotation vectors

def compute_angular_velocity(delta_rot, dt):
    """
    Compute angular velocity from two rotation matrices

    delta_rot =  exp(omega * dt)

    
    """
    # angular_velocity = np.zeros((3,))
    # delta_rot = C1.T @C2
    if delta_rot.shape[0] == 3:
        # For SO3
        angular_velocity = SO3.Log(delta_rot).ravel() / dt
    elif delta_rot.shape[0] == 2:
        # For SO2
        angular_velocity = SO2.Log(delta_rot) / dt
    return angular_velocity


def compute_error(ground_truth, estimated, times_est, dt):
    errors = []
    outlier = 0
    mag_motion = []
    z = 0
    roll = 0
    pitch = 0
    for i, t in enumerate(times_est):
        if i > 0:
            # if dt is too large, skip the pose
            # if t - times_fou[i-1] > 0.5:
            #     continue
            tk_1 = t - dt
            Ti = state_interp(tk_1, ground_truth, method="linear").value
            Tj = state_interp(t, ground_truth, method="linear").value
            Tij_gt = SE3.inverse(Ti) @ Tj 
            Tij_fou = estimated[i]
            error_pose = SE3.inverse(Tij_gt) @ Tij_fou
            error_rot = SO3.Log(error_pose[:3, :3]).ravel()
            error_trans = error_pose[:3, 3]

            error_trans = Tij_gt[:3, 3] - Tij_fou[:3, 3]

            vec_rel_pose = SE3.Log(Tij_gt).ravel()

            # Filter out large motions in roll pitch and z
            if abs(vec_rel_pose[0]) > np.deg2rad(20):
                roll += 1
                continue
            if abs(vec_rel_pose[1]) > np.deg2rad(20):
                pitch += 1
                continue
            if abs(Tij_gt[2, 3]) > 0.3:
                z += 1
                continue

            mag_motion.append([np.linalg.norm(vec_rel_pose[2]), np.linalg.norm(Tij_gt[:2, 3])])

            # if vec_rel_pose[0] > np.deg2rad(20) or vec_rel_pose[1] > np.deg2rad(20) or Tij_gt[2, 3] > 0.3:
            #     continue

            # if error_rot[2] > np.deg2rad(5) or np.linalg.norm(error_trans) > 0.2:
            #     outlier += 1
            #     continue
            errors.append(np.block([[error_rot], [error_trans]]).ravel())

    print("Large roll pitch and z motions filtered:", roll, pitch, z)
    mag_motion = np.array(mag_motion)
    mag_motion_mean = np.mean(mag_motion, axis=0)
    print("Mean motion (rad, m):", np.rad2deg(mag_motion_mean[0]), mag_motion_mean[1])
    return errors



# Find outliers and remove them
def remove_outliers(errors, threshold_rot = np.deg2rad(5), threshold_trans = 0.5):
    # Find outliers
    outliers = []
    N_total = errors.shape[0]
    for i in range(errors.shape[0]):
        error_rot = errors[i, :3]
        error_trans = errors[i, 3:]
        if error_rot[2] > threshold_rot or np.linalg.norm(error_trans) > threshold_trans:
            outliers.append(i)
    # Remove outliers
    print(f"Removing {len(outliers)} outliers")
    return np.delete(errors, outliers, axis=0), len(outliers) / N_total * 100.0


C_cs = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]], dtype=np.float32) @ SO3.Exp(np.array([0, np.deg2rad(-15), 0], dtype=np.float32))

r_c = 1e-3 *np.array([-22.189, 130.691, 78.172], dtype=np.float32)

T_cs = np.eye(4, dtype=np.float32)
T_cs[:3, :3] = C_cs[:3, :3]
T_cs[:3, 3] = r_c

T_cs_se3 = SE3State(T_cs)

# eye_se3 = SE3State(np.eye(4, dtype=np.float32))


# plot_poses([T_cs_se3, eye_se3], step = 1, arrow_length=0.1)
# plt.show()


T_cb = np.array([
    [-0.9999933560636873, 0.0024426354777743113, 0.0027058012505205144, 0.010014539463188655],
    [-0.0024423854745165723, -0.9999970127932984, 9.569574046289494e-05, -0.0032371785334024787],
    [0.0027060269175436104, 8.908649499528979e-05, 0.9999963347342424, -0.08921631720790653],
    [0.0, 0.0, 0.0, 1.0]])
T_bc = SE3.inverse(T_cb)

T_bs = T_bc @ T_cs
C_bs = T_bs[:3, :3]


dataset = "/media/arturo/Voyis/Quarry0211/truck_lm/"

gt = dataset + "cameras.xml"

polar_images = dataset + "sonar/"
data_path = Path(polar_images)
gt_path = Path(gt)

# Load states
import yaml
config_dir = "/home/arturo/Documents/voyis_tools/feature_matching/config"
config_dir = Path(config_dir)
with open(config_dir.joinpath("config_camera.yaml")) as f:
    next(f)
    cam_param: dict = yaml.load(f, Loader=yaml.FullLoader)

    ## TODO correct timestamps
gt_states = readAgisoftTrajectory(gt, cam_param)

gt_states_sonar = [ ]
for state in gt_states:
    state_s = state.copy()
    state_s.value = state.value @ T_bs
    gt_states_sonar.append(state_s)

# cam_file_csv = "/home/arturo/data/Quarry0211/truck_lm/csv/Discovery_stereo-Camera.csv"
# gt_states = correct_timestamps(gt_states, cam_file_csv)

# bspline = SE3Bspline(gt_states, max_dt = 1.0/3.0)
# gt_states = [gt_states[0].minus(state) for state in gt_states]  # Interpolate states to match timestamps

### From 0 to 200, the ROV was moving for alignment.
# Right at 100, it starts yawing.



# print(datetime.utcfromtimestamp(gt_states[0].stamp).strftime('%Y-%m-%d %H:%M:%S'))

# fig, ax = plot_poses(gt_states, step = 10) # gt poses look good
# plt.show()

rel_poses_gt = []
for i in range(len(gt_states_sonar) - 1):
    T_i = gt_states_sonar[i].value
    T_j = gt_states_sonar[i + 1].value
    Tij_gt = SE3.inverse(T_i) @ T_j
    rel_poses_gt.append(Tij_gt)

times_gt = np.array([state.stamp for state in gt_states_sonar])
rot_vec = [ SO3.Log(state[:3, :3]).ravel()  for state in rel_poses_gt]
rot_vec = np.vstack(rot_vec)
trans_vec = np.vstack([state[:3, 3].ravel() for state in rel_poses_gt])
fig, axs = plt.subplots(3, 1)
# axs[0].plot(pd.to_datetime(times_gt[20:200], unit='s'), rot_vec[20:200, 0])
# axs[1].plot(pd.to_datetime(times_gt[20:200], unit='s'), rot_vec[20:200, 1])
axs[0].plot(pd.to_datetime(times_gt[20:200], unit='s'), rot_vec[20:200, 2])
axs[1].plot(pd.to_datetime(times_gt[20:200], unit='s'), trans_vec[20:200, 0])
axs[2].plot(pd.to_datetime(times_gt[20:200], unit='s'), trans_vec[20:200, 1])
fig.suptitle('Ground Truth Relative Pose Vectors')





skips = [5, 15, 30, 60, 100]
# skips = [1, 5]
ang_outlier_threshold = [np.deg2rad(10), np.deg2rad(10), np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)] # 10
trans_outlier_threshold = [0.3, 0.3, 0.3, 0.3, 0.3] # 0.3
all_icp_errors = []
all_dir_errors = []
all_fou_errors = []
all_rad_errors = []
all_teas_errors = []
all_init_errors = []


icp_outliers = []
dir_outliers = []
fou_outliers = []
rad_outliers = []
teas_outliers = []

for k, skip in enumerate(skips):

    folder = "/home/arturo/data/Quarry0211/truck_lm/skip" + str(skip)



    toff = 21407524.741737127 # time offset known from feb quarry dataset
    toff-= 0.02



    gamma = 1305 / (
                        2.0 * 10.0 * np.sin(np.deg2rad(130) / 2.0)
                    )
    """
    Load results from the specified run and folder.
    """
    icp_results = np.load(f"{folder}/icp_results.npz")

    fou_results = np.load(f"{folder}/fourier_results.npz")

    dir_results = np.load(f"{folder}/direct_results.npz")
    teaser_results = np.load(f"{folder}/teaser_results.npz")
    icp_poses = icp_results['icp_se3']
    fou_poses = fou_results['fou_se3']
    dir_poses = dir_results['dir_se3']
    teaser_poses = teaser_results['teas_se3']


    # Remove nans and count them
    icp_poses = icp_poses[~np.isnan(icp_poses).any(axis=(1))]
    fou_poses = fou_poses[~np.isnan(fou_poses).any(axis=(1))]
    dir_poses = dir_poses[~np.isnan(dir_poses).any(axis=(1))]
    teaser_poses = teaser_poses[~np.isnan(teaser_poses).any(axis=1)]

    # fou_poses[:, 3:] /= gamma  # Scale the shift by gamma

    times_icp = icp_results['times']
    times_fou = fou_results['times']
    times_dir = dir_results['times']
    times_teas = teaser_results['times']

    try:
        rad_results = np.load(f"{folder}/radon_results.npz")
        rad_poses = rad_results['rad_se3']
        times_rad = rad_results['times']
        times_rad = times_rad + toff
    except:
        rad_poses = None
        times_rad = None




    # /delta_t_init_fou = gyro_data['#timestamp'][0] - times_fou[0]   # Initial time offset to align the first sonar timestamp with the first gyro timestamp
    ## overlap angular velocities from sonar with IMU data
    times_fou = times_fou + toff  # Adjusting for a known offset in the dataset

    # delta_t_init_icp = gyro_data['#timestamp'][0] - times_icp[0]   # Initial time offset to align the first sonar timestamp with the first gyro timestamp
    ## overlap angular velocities from sonar with IMU data
    times_icp = times_icp + toff 

    # delta_t_init_dir = gyro_data['#timestamp'][0] - times_dir[0]   # Initial time offset to align the first sonar timestamp with the first gyro timestamp
    ## overlap angular velocities from sonar with IMU data
    times_dir = times_dir + toff

    times_teaser = times_teas + toff
    # delta_t_init_rad = gyro_data['#timestamp'][0] - times_rad[0]   # Initial time offset to align the first sonar timestamp with the first gyro timestamp
    ## overlap angular velocities from sonar with IMU data


    rel_poses_icp = []
    ang_velocities_icp = []
    for i in range(icp_poses.shape[0] // 4):
        rel_poses_icp.append(icp_poses[i * 4: (i + 1) * 4])
    # rel_poses_icp = [T_cs @ T @ SE3.inverse(T_cs) for T in rel_poses_icp]
    rot_vector_icp = np.array([SO3.Log(T[:3, :3]).ravel() for T in rel_poses_icp])
    trans_vector_icp = np.array([T[:3, 3] for T in rel_poses_icp])


    rel_poses_teas = []
    ang_velocities_teas = []
    for i in range(teaser_poses.shape[0] // 4):
        rel_poses_teas.append(teaser_poses[i * 4: (i + 1) * 4])
    # rel_poses_icp = [T_cs @ T @ SE3.inverse(T_cs) for T in rel_poses_icp]
    rot_vector_teas = np.array([SO3.Log(T[:3, :3]).ravel() for T in rel_poses_teas])
    trans_vector_teas = np.array([T[:3, 3] for T in rel_poses_teas])

    rel_poses_dir = []
    ang_velocities_dir = []
    for i in range(dir_poses.shape[0] // 3):
        se2_pose = dir_poses[i * 3: (i + 1) * 3]
        so3_rot = SO3.Exp(np.array([0, 0, SO2.Log(se2_pose[:2, :2])]))
        shift = np.array([se2_pose[0, 2], se2_pose[1, 2], 0], dtype=np.float32) 
        se3_pose = SE3.from_components(so3_rot, shift)
        rel_poses_dir.append(se3_pose)
    # rel_poses_dir = [T_cs @ T @ SE3.inverse(T_cs) for T in rel_poses_dir]
    rot_vector_dir = np.array([SO3.Log(T[:3, :3]).ravel() for T in rel_poses_dir])
    trans_vector_dir = np.array([T[:3, 3] for T in rel_poses_dir])

    rel_poses_fou = []
    ang_velocities_fou = []
    for i in range(fou_poses.shape[0] // 3):
        se2_pose = fou_poses[i * 3: (i + 1) * 3]
        so3_rot = SO3.Exp(np.array([0, 0, SO2.Log(se2_pose[:2, :2])]))
        shift = np.array([se2_pose[0, 2], se2_pose[1, 2], 0], dtype=np.float32) / gamma
        se3_pose = SE3.from_components(so3_rot, shift)
        rel_poses_fou.append(se3_pose)
        
    # rel_poses_fou = [T_cs @ T @ SE3.inverse(T_cs) for T in rel_poses_fou]
    rot_vector_fou = np.array([SO3.Log(T[:3, :3]).ravel() for T in rel_poses_fou])
    trans_vector_fou = np.array([T[:3, 3] for T in rel_poses_fou])

    # Init poses (zeros)
    rel_init_poses = []
    for i in range(fou_poses.shape[0] // 3):
        se3_pose = np.eye(4, dtype=np.float32)
        rel_init_poses.append(se3_pose)

    if rad_poses is not None:
        rel_poses_rad = []
        ang_velocities_rad = []
        for i in range(rad_poses.shape[0] // 3):
            se2_pose = rad_poses[i * 3: (i + 1) * 3]
            so3_rot = SO3.Exp(np.array([0, 0, SO2.Log(se2_pose[:2, :2])]))
            shift = np.array([se2_pose[0, 2], se2_pose[1, 2], 0], dtype=np.float32) / gamma
            se3_pose = SE3.from_components(so3_rot, shift)
            rel_poses_rad.append(se3_pose)
        # rel_poses_fou = [T_cs @ T @ SE3.inverse(T_cs) for T in rel_poses_fou]
        rot_vector_rad = np.array([SO3.Log(T[:3, :3]).ravel() for T in rel_poses_rad])
        trans_vector_rad = np.array([T[:3, 3] for T in rel_poses_rad])

    import matplotlib.pyplot as plt

    #Plot the rotation vectors
    # fig, axs = plt.subplots(4, 1, figsize=(10, 5))
    # axs[0].plot(pd.to_datetime(times_dir, unit='s'), rot_vector_dir[:, 2], label='Direct Rotation Z')
    # axs[1].plot(pd.to_datetime(times_fou, unit='s'), rot_vector_fou[:, 2], label='Fourier Rotation Z')
    # axs[2].plot(pd.to_datetime(times_icp, unit='s'), rot_vector_icp[:, 2], label='ICP Rotation Z')
    # axs[3].plot(pd.to_datetime(times_rad, unit='s'), rot_vector_rad[:, 2], label='Radon Rotation Z')
    # fig.suptitle('Relative Rotation Vectors from Sonar Registration')
    # fig.supxlabel('Time (s)')
    # fig.supylabel('Yaw (rad)')
    # fig.legend()

    # # fig.grid()

    # # Plot the x translations464033
    # fig, axs = plt.subplots(4, 1, figsize=(10, 5))
    # axs[0].plot(pd.to_datetime(times_dir, unit='s'), trans_vector_dir[:, 0], label='Direct Translation X')
    # axs[1].plot(pd.to_datetime(times_fou, unit='s'), trans_vector_fou[:, 0], label='Fourier Translation X')
    # axs[2].plot(pd.to_datetime(times_icp, unit='s'), trans_vector_icp[:, 0], label='ICP Translation X')
    # axs[3].plot(pd.to_datetime(times_rad, unit='s'), trans_vector_rad[:, 0], label='Radon Translation X')
    # fig.suptitle('Relative Translations from Sonar Registration')
    # fig.supxlabel('Time (s)')
    # fig.supylabel('X (m)')
    # fig.legend()



    # fig, axs = plt.subplots(4, 1, figsize=(10, 5))
    # axs[0].plot(pd.to_datetime(times_dir, unit='s'), ang_velocities_dir[:, 2], label='Direct Rotation Z')
    # axs[1].plot(pd.to_datetime(times_fou, unit='s'), ang_velocities_fou[:, 2], label='Fourier Rotation Z')
    # axs[2].plot(pd.to_datetime(times_icp, unit='s'), ang_velocities_icp[:, 2], label='ICP Rotation Z')
    # axs[3].plot(pd.to_datetime(times_rad, unit='s'), ang_velocities_rad[:, 2], label='Radon Rotation Z')
    # fig.suptitle('Angular Velocities from Sonar Registration')
    # fig.supxlabel('Time (s)')
    # fig.supylabel('Yaw velocity (rad / s)')
    # fig.legend()





    # plt.show()

    # plt.plot(times_gt)





    # for frequency of 3
    # times_fou = times_fou[20:200]
    # times_icp = times_icp[20:200]
    # times_dir = times_dir[20:200]

    # for frequency of 15
    # times_fou = times_fou[100:1000]
    # times_icp = times_icp[100:1000]
    # times_dir = times_dir[100:1000]


    errors_fou = compute_error(gt_states_sonar, rel_poses_fou, times_fou, dt = skip * (1.0/15.0))
    errors_dir = compute_error(gt_states_sonar, rel_poses_dir, times_dir, dt = skip * (1.0/15.0))
    errors_icp = compute_error(gt_states_sonar, rel_poses_icp, times_icp, dt = skip * (1.0/15.0))
    errors_teas = compute_error(gt_states_sonar, rel_poses_teas, times_teaser, dt = skip * (1.0/15.0))

    errors_init = compute_error(gt_states_sonar, rel_init_poses, times_fou, dt = skip * (1.0/15.0))
    errors_init = np.array(errors_init)

    errors_fou = np.array(errors_fou)
    error_fou_m = np.mean(errors_fou, axis=0)
    error_fou_std = np.std(errors_fou, axis=0)

    print(f"Mean Fourier error", error_fou_m)
    print(f"Std Fourier error", error_fou_std)






    errors_icp = np.array(errors_icp)
    error_icp_m = np.mean(errors_icp, axis=0)
    error_icp_std = np.std(errors_icp, axis=0)

    print(f"Mean ICP error", error_icp_m)
    print(f"Std ICP error", error_icp_std)


    errors_dir = np.array(errors_dir)
    error_dir_m = np.mean(errors_dir, axis=0)
    error_dir_std = np.std(errors_dir, axis=0)

    print(f"Mean Direct error", error_dir_m)
    print(f"Std Direct error", error_dir_std)

    errors_teas = np.array(errors_teas)
    error_teas_m = np.mean(errors_teas, axis=0)
    error_teas_std = np.std(errors_teas, axis=0)

    print(f"Mean TEASER error", error_teas_m)
    print(f"Std TEASER error", error_teas_std)

    
    errors_rad = compute_error(gt_states_sonar, rel_poses_rad, times_rad, dt= skip * (1.0/15.0))
    errors_rad = np.array(errors_rad)
    error_rad_m = np.mean(errors_rad, axis=0)
    error_rad_std = np.std(errors_rad, axis=0)
    error_sq_rad = np.sqrt(errors_rad**2)
    
    print(f"Mean Radon error", error_rad_m)
    print(f"Std Radon error", error_rad_std)

    # Violin plots
    error_sq_icp = np.sqrt(errors_icp**2)
    error_sq_fou = np.sqrt(errors_fou**2)
    error_sq_dir = np.sqrt(errors_dir**2)
    error_sq_teas = np.sqrt(errors_teas**2)
    error_sq_init = np.sqrt(errors_init**2)

    # At this stage, I want to count the number of outliers in each method,
    # remove them and then plot the boxplots again

    ang_t = ang_outlier_threshold[k]
    trans_t = trans_outlier_threshold[k]

    error_sq_icp, icp_out = remove_outliers(error_sq_icp, threshold_rot=ang_t, threshold_trans=trans_t)
    error_sq_fou, fou_out = remove_outliers(error_sq_fou, threshold_rot=ang_t, threshold_trans=trans_t)
    error_sq_dir, dir_out = remove_outliers(error_sq_dir, threshold_rot=ang_t, threshold_trans=trans_t)
    error_sq_teas, teas_out = remove_outliers(error_sq_teas, threshold_rot=ang_t, threshold_trans=trans_t)
    error_sq_rad, rad_out = remove_outliers(error_sq_rad, threshold_rot=ang_t, threshold_trans=trans_t)

    icp_outliers.append(icp_out)
    fou_outliers.append(fou_out)
    dir_outliers.append(dir_out)
    teas_outliers.append(teas_out)
    rad_outliers.append(rad_out)

    all_icp_errors.append(error_sq_icp)
    all_fou_errors.append(error_sq_fou)
    all_dir_errors.append(error_sq_dir)
    all_teas_errors.append(error_sq_teas)
    all_rad_errors.append(error_sq_rad)
    all_init_errors.append(error_sq_init)


  

# # Plot the errors
# fig, axs = plt.subplots(3, 1, figsize=(25, 15))
# axs[0].plot(pd.to_datetime(times_fou[1:], unit='s'), np.rad2deg(errors_fou[:, 2]), label='Fourier', linewidth=2)
# axs[0].plot(pd.to_datetime(times_dir[1:], unit='s'), np.rad2deg(errors_dir[:, 2]), label='Direct', linewidth=2)
# axs[0].plot(pd.to_datetime(times_icp[1:], unit='s'), np.rad2deg(errors_icp[:, 2]), label='ICP', linewidth=2)
# # axs[0].plot(pd.to_datetime(times_teaser[1:], unit='s'), np.rad2deg(errors_teas[:, 2]), label='TEASER', linewidth=2)
# axs[1].plot(pd.to_datetime(times_fou[1:], unit='s'), errors_fou[:, 3],  linewidth=2)
# axs[1].plot(pd.to_datetime(times_dir[1:], unit='s'), errors_dir[:, 3],  linewidth=2)
# axs[1].plot(pd.to_datetime(times_icp[1:], unit='s'), errors_icp[:, 3], linewidth=2)
# # axs[1].plot(pd.to_datetime(times_teaser[1:], unit='s'), errors_teas[:, 3], linewidth=2)

# axs[2].plot(pd.to_datetime(times_fou[1:], unit='s'), errors_fou[:, 4],  linewidth=2)
# axs[2].plot(pd.to_datetime(times_dir[1:], unit='s'), errors_dir[:, 4],  linewidth=2)
# axs[2].plot(pd.to_datetime(times_icp[1:], unit='s'), errors_icp[:, 4],  linewidth=2)
# # axs[2].plot(pd.to_datetime(times_teaser[1:], unit='s'), errors_teas[:, 4],  linewidth=2)
# # 
# if rad_poses is not None:
#     axs[0].plot(pd.to_datetime(times_rad[1:], unit='s'), np.rad2deg(errors_rad[:, 2]), label='Radon', linewidth=2)
#     axs[1].plot(pd.to_datetime(times_rad[1:], unit='s'), errors_rad[:, 3], linewidth=2)
#     axs[2].plot(pd.to_datetime(times_rad[1:], unit='s'), errors_rad[:, 4],  linewidth=2)
# fig.suptitle('Sonar Registration Errors', size = 30)
# fig.supxlabel('Time (s)', size = 25)
# # fig.supylabel('Errors', size = 25)
# fig.legend(fontsize=25, loc='upper right')
# axs[0].set_ylabel('Yaw Error [deg]', size = 25)
# axs[1].set_ylabel('X Error [m]', size = 25)
# axs[2].set_ylabel('Y Error [m]', size = 25)
# # Set tick sizes for each axis
# axs[0].tick_params(axis='both', which='major', labelsize=20)
# axs[1].tick_params(axis='both', which='major', labelsize=20)
# axs[2].tick_params(axis='both', which='major', labelsize=20)

# plt.tight_layout()
# plt.show()  





# fig, ax = plt.subplots(3,1)

# if rad_poses is not None:
#     for i in range(3):

#         ax[i].boxplot([error_sq_icp[:, i+2], error_sq_fou[:, i+2], error_sq_dir[:, i+2],error_sq_rad[:, i+2]],
#                 showmeans=True)
#         ax[i].set_xticks([1, 2, 3, 4])
#         ax[i].set_xticklabels([r'ICP', r'Fourier', r'Direct', r'Radon'])
# else:
#     for i in range(3):
#         ax[i].boxplot([error_sq_icp[:, i+2], error_sq_fou[:, i+2], error_sq_dir[:, i+2], error_sq_teas[:, i+2]],
#                 showmeans=True)
#         ax[i].set_xticks([1, 2, 3, 4])
#         ax[i].set_xticklabels([r'ICP', r'Fourier', r'Direct', r'TEASER'])
# ax[0].set_ylabel(r'$\delta\alpha \, [\mathrm{{rad}}]$')
# ax[1].set_ylabel(r'$\delta x \,[\mathrm{{m}}]$')
# ax[2].set_ylabel(r'$\delta y \, [\mathrm{{m}}]$')
# ax[2].set_xlabel(r'Method')
# ax[0].set_title(r'Registration errors on Quarry segment')
# plt.tight_layout()
# plt.show()


import seaborn as sns
sns.set_theme(style="whitegrid", palette="colorblind")

methods = [r'ICP', r'FT', r'Direct', r'RT']
method_errors = [all_icp_errors, all_fou_errors, all_dir_errors, all_rad_errors]
error_labels = [r'$\delta\alpha$', r'$\delta x$', r'$\delta y$']
ylabels = [r'$\delta\alpha \, [\mathrm{rad}]$', r'$\delta x \, [\mathrm{m}]$', r'$\delta y \, [\mathrm{m}]$']

fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

for i in range(3):  # 0: yaw, 1: x, 2: y
    records = []
    all_vals = []
    for method, method_name in zip(method_errors, methods):
        for skip, errors in zip(skips, method):
            vals = errors[:, 2 + i]  # 2: yaw, 3: x, 4: y
            all_vals.append(vals)
            for val in errors[:, 2 + i]:  # 2: yaw, 3: x, 4: y
                records.append({'Method': method_name, 'Skip': skip, 'Error': val})
    df = pd.DataFrame(records)
    sns.boxplot(x='Skip', y='Error', hue='Method', data=df, ax=axes[i], width=0.8, showfliers=False)
    axes[i].set_ylabel(ylabels[i])
    # axes[i].set_title(f'{error_labels[i]} error')
    if i == 0:
        handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend_.remove()  # Remove individual legends

    # Compute the largest 4th quartile (75th percentile) across all data for this error type
#     q3 = max(np.percentile(vals, 93) for vals in all_vals)
#     axes[i].set_ylim(top=q3, bottom=-0.0001)  # Set the same y-axis limit for all subplots

axes[0].set_ylim(top= 0.05, bottom=-0.001)  # 0.05  #0.25 yaw
axes[1].set_ylim(top= 0.09, bottom=-0.001)  # 0.09 #0.2 yaw
axes[2].set_ylim(top= 0.15, bottom=-0.001)  # 0.15 #0.4 yaw
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, .97),ncol=len(methods), frameon=True)
# fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, 1.02))
axes[-1].set_xlabel(r'Frames skipped between registrations')


# fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, 1.02))
# axes[-1].set_xlabel(r'Frames skipped between registrations')

# fig.suptitle(r'Registration errors on real data')
plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space at the top for the legend
# plt.tight_layout()
# plt.show()




### Bar plot of outliers

# Prepare data
# methods = [r'ICP', r'Fourier', r'Direct', r'Radon']
outlier_lists = [icp_outliers, fou_outliers, dir_outliers, rad_outliers]

records = []
for method, outlier_counts in zip(methods, outlier_lists):
    for skip, count in zip(skips, outlier_counts):
        records.append({'Method': method, 'Skip': skip, 'Outliers': count})

df_outliers = pd.DataFrame(records)

plt.figure(figsize=(4, 6))
sns.barplot(x='Skip', y='Outliers', hue='Method', data=df_outliers)
plt.ylabel(r'Percentage of Outliers')
plt.xlabel(r'Frames skipped between registrations')
plt.title(r'Percentage of Outliers per Method and Skip')
plt.ylim(0, 100)
plt.legend(title='Method', loc='upper left')
plt.tight_layout()
plt.show()

# Make a latex tabular table with the number of outliers per method and skip
print("Outlier percentages:")
print(r"\begin{tabular}{lccccc}")
print(r"Method & Skip 5 & Skip 15 & Skip 30 & Skip 60 & Skip 100 \\ \hline")
for method, outlier_counts in zip(methods, outlier_lists):
    outlier_percentages = [f"{count:.1f}\%" for count in outlier_counts]
    print(f"{method} & " + " & ".join(outlier_percentages) + r" \\")
print(r"\end{tabular}")