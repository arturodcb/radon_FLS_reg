import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from pathlib import Path
import time


from image_reg import ICPRegistration, FourierRegistration, DirectRegistration, RadonRegistration

import ciso8601

import FLS
from utils import plot_polar, to_cartesian, load_gt, convert_se3_estimate_to_planar_estimates, state_interp


# def plot_tags_polar(data: np.ndarray, fov_degrees: float, min_range: float, max_range: float, detected_tags, title: str='') -> None:
#     # Optimal tag corner text alignments (used for plotting later)
#     corner_ha_alignments = ['left', 'left', 'right', 'right']
#     corner_va_alignments = ['top', 'bottom', 'bottom', 'top']
    
#     # Get theta and range values for the image
#     min_theta, max_theta = np.array([-fov_degrees/2, fov_degrees/2])
#     ranges = np.linspace(max_range, min_range, data.shape[0])
#     thetas = np.deg2rad(np.linspace(max_theta, min_theta, data.shape[1]))

#     # Create a meshgrid so we can plot the 2D function
#     thetasMesh, rangesMesh = np.meshgrid(thetas, ranges)
    
#     # Create polar plot and set image values.
#     figp, axp = plt.subplots(subplot_kw=dict(projection='polar'))
#     axp.pcolormesh(thetasMesh, rangesMesh, data)

#     # Adjust origin to be in the center of the image
#     axp.set_theta_zero_location('N')

#     # Set bounds for theta and range
#     axp.set_thetamin(min_theta)
#     axp.set_thetamax(max_theta)
#     axp.set_rmax(max_range)
#     axp.set_theta_direction(-1)

#     # Set title and turn off the grid
#     axp.set_title(title)
#     # axp.grid(False)
#     plt.show()



dataset = "/media/arturo/Voyis/OceanSIM/welland_boat_planar/"

gt = dataset + "gt_states.csv"

polar_images = dataset + "sonar/"
data_path = Path(polar_images)
gt_path = Path(gt)

# Load states
gt_states = load_gt(gt_path)


prev_image = None
prev_image_polar = None
prev_image_name = None
prev_pose = None

r_min = 0.2
r_max = 3.0
azimuth_fov = 130
elevation_fov = 20


# FLSmodel = FLS.Model(r_min = 0, r_max = 13, azimuth_fov = 130, elevation_fov = 20,
#            width_image = 1305)




icp_reg = ICPRegistration(r_max = 3, r_min = 0, azimuth_fov = 130, elevation_fov = 20)
fourier_reg = FourierRegistration(r_max = 3, r_min = 0, azimuth_fov = 130, elevation_fov = 20)
dir_reg = DirectRegistration(r_max = 3, r_min = 0, azimuth_fov = 130, elevation_fov = 20)
rad_reg = RadonRegistration(r_max = 3, r_min = 0, azimuth_fov = 130, elevation_fov = 20)
# pcd = o3d.t.geometry.PointCloud(o3c.Device("cuda:0"))

from pymlg.numpy import SE3, SO3, SE2, SO2
from states import SE3State

ext_rot = SO3.Exp(np.array([np.pi, 0, 0], dtype=np.float32))  # 10 degrees rotation around y-axis
ext_trans = np.array([0.3, 0, 0.3], dtype=np.float32)  # Translation in x and z axes
T_bs = SE3.from_components(ext_rot, ext_trans)
T_bs_se3 = SE3State(T_bs)

sonar_poses = [T_ab.plus(T_bs_se3) for T_ab in gt_states]

# fig, ax = plot_poses(sonar_poses, step=100)
# plt.show()

# gt_se2 = []
# icp_se2 = []
# fou_se2 = []
# dir_se2 = []
# initial_se2 = []


gt_se3 = []
dir_se3 = []
initial_se3 = []
icp_se3 = []
fou_se3 = []
rad_se3 = []
teas_se3 = []

n_img = 0
for polar_file in sorted(data_path.iterdir(), key=lambda x: x.name):
    
        #print("Processing image: ", polar_file.name)
    if n_img % 5 == 0 and n_img >50 :

        polar_img = cv2.imread(str(polar_file), cv2.IMREAD_GRAYSCALE)

        ts = polar_file.stem.split('_')[-2]  # Assuming the timestamp is in the filename
        

        # convert datetime to epoch time
        ts_converted = ciso8601.parse_datetime(ts.replace("-", "")).timestamp()
            
        T_as = state_interp(ts_converted, sonar_poses).value
        
        image = to_cartesian(polar_img, min_range=r_min, max_range=r_max, azimuth_fov=azimuth_fov)
        
        if image.max() == 0:
            print(f"Skipping image {polar_file.name} due to it being all zeros.")
            continue
        

        # fan_mask = create_fan_mask(image.shape, 25)
        # # rad = cv2.cvtColor(radon(image), cv2.COLOR_GRAY2BGR)
        # image_with_white_bg = image.copy()
        # image_with_white_bg[fan_mask == 0] = 255  #
        # # image_with_white_bg = image.copy()
        # # image_with_white_bg[~fan_mask] = 255
        # cv2.imshow("image with white bg", image_with_white_bg)
        # cv2.waitKey(0)
        #         # fan_mask = cr
        # Check this, do I need to account for r_min? ans: NO
        gamma = image.shape[1] / (
                    2.0 * r_max * np.sin(np.deg2rad(azimuth_fov) / 2.0)
                )

        if prev_image is not None:

            print("Processing image: ", Path(polar_file).name)
            print("Reference image: ", prev_image_name)

            T_ij = SE3.inverse(prev_pose) @ T_as # This should be good, warp takes T_ij as input

            vector = SE3.Log(T_ij).ravel()

            # print(f"Relative pose vector: {vector}")

            ang_gt = np.rad2deg(vector[2])
            shift_gt = np.array([T_ij[0, 3], T_ij[1, 3]], dtype=np.float32) 
            print("Angle (degrees):", ang_gt)
            print("Shift (meters):", shift_gt )
            print("Shift (pixels):", shift_gt * gamma)
            print()
            # gt_se2.append(np.array([ang_gt, shift_gt[0], shift_gt[1]], dtype=np.float32))
            gt_se3.append(np.block([[SO3.Log(T_ij[:3, :3]).ravel()], [T_ij[:3, 3]]]).ravel())

            # Add noise to ground truth SE2 transformation
            T_ab_SE2 = np.eye(3, dtype=np.float32)
            T_ab_SE2[:2, :2] = SO2.Exp(vector[2])
            T_ab_SE2[:2, 2] = shift_gt       
            T_ab_SE2_noisy = T_ab_SE2 @ SE2.Exp(np.random.normal(0, 7e-2, size=3))  # Add some noise to the transformation
            noisy_shift = np.array([T_ab_SE2_noisy[0, 2], T_ab_SE2_noisy[1, 2]], dtype=np.float32)
            noisy_angle = np.rad2deg(SO2.Log(T_ab_SE2_noisy[:2, :2]))
            
            T_ij_noisy_se3 = SE3.from_components(SO3.Exp(np.array([0, 0, np.deg2rad(noisy_angle)])),
                                                  np.array([noisy_shift[0], noisy_shift[1], 0], dtype=np.float32))

            initial_se3.append(np.block([[SO3.Log(T_ij_noisy_se3[:3, :3]).ravel()], [T_ij_noisy_se3[:3, 3]]]).ravel())

            # Add noise: More in XYZ Yaw and less in RP -> IMU Deadreckon
            # xy_yaw_noise = np.random.normal(0, 3e-2, size=3)  # Small noise in XYZ and Yaw
            # rpz_noise = np.random.normal(0, 0, size=2)  # Smaller noise in RP
            # rpz_noise = np.zeros(3)
            # T_ij_n = T_ij @ SE3.Exp(np.array([rpz_noise[0], rpz_noise[1], xy_yaw_noise[0], xy_yaw_noise[1], xy_yaw_noise[2], rpz_noise[2]], dtype=np.float32))
            # noisy_angle_n = np.rad2deg(SO3.Log(T_ij_n[:3, :3])[2][0])
            # noisy_shift_n = np.array([T_ij_n[0, 3], T_ij_n[1, 3]], dtype=np.float32)

            # T_ij_n_se2 = SE2.from_components(SO2.Exp(np.deg2rad(noisy_angle_n)),
            #                                       np.array([noisy_shift_n[0], noisy_shift_n[1]], dtype=np.float32))


            # initial_se2.append(np.array([noisy_angle, noisy_shift[0], noisy_shift[1]], dtype=np.float32))
            
            # print(f"Initial guess vector: {SE3.Log(T_ij_n).ravel()}")

            # print(T_ij_n)

            # print(f"Error vector: ", SE3.Log(SE3.inverse(T_ij_n) @ T_ij))
            # print("Initial angle (degrees):", noisy_angle_n)
            # print("Initial shift (meters):", noisy_shift_n)
            # print("Initial shift (pixels):", noisy_shift_n * gamma)
            # print()
            # print(f"Error initial guess (degrees): {np.abs(ang_gt - noisy_angle)}")
            # print(f"Error initial guess (m): {np.linalg.norm(shift_gt - noisy_shift)}")
            # print()


            # Visualize the initial guess
            # SonarImageRegistration.visualize_registration(prev_image.copy(), image.copy(), noisy_angle_n, gamma * noisy_shift_n, "Initial Guess")

            ## ICP REGISTRATION
            try:
                time_start = time.time()
                pose = icp_reg.register_images(prev_image, image, prev_image_polar, polar_img, None)
                time_end = time.time()
                print(f"ICP Registration took {time_end - time_start:.2f} seconds")
                ang, shift = convert_se3_estimate_to_planar_estimates(pose)
                # ICPRegistration.visualize_registration(prev_image, image, ang, gamma*shift, "ICP Registration Result")
                print(f"[ICP] Pose vector: {SE3.Log(pose).ravel()}")
                # icp_se2.append(np.array([ang, shift[0], shift[1]], dtype=np.float32))
                icp_se3.append(np.block([[SO3.Log(pose[:3, :3]).ravel()], [pose[:3, 3]]]).ravel())
            except Exception as e:
                print(f"ICP Registration failed for {polar_file.name}: {e}")
                icp_se3.append(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32))
            # ### FOURIER REGISTRATION
            try:
                time_start = time.time()
                pose = fourier_reg.register_images(prev_image, image, prev_image_polar, polar_img, None)
                time_end = time.time()
                print(f"Fourier Registration took {time_end - time_start:.2f} seconds")
                ang = np.rad2deg(SO2.Log(pose[:2, :2]))
                shift = np.array([pose[0, 2], pose[1, 2]], dtype=np.float32)
                # ang, shift = convert_se3_estimate_to_planar_estimates(pose)
                # FourierRegistration.visualize_registration(prev_image, image, ang, shift, "Fourier Registration Result")
                # fou_se2.append(np.array([ang, shift[0], shift[1]], dtype=np.float32))
                

                fou_se3.append(np.array([0, 0, SO2.Log(pose[:2, :2]), shift[0], shift[1], 0], dtype=np.float32))
            except Exception as e:
                print(f"Fourier Registration failed for {polar_file.name}: {e}")
            #     # fou_se2.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                fou_se3.append(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32))

            # ### DIRECT REGISTRATION
            try:
                time_start = time.time()
                pose = dir_reg.register_images(prev_image, image, None, None,  None )
                time_end = time.time()
                print(f"Direct Registration took {time_end - time_start:.2f} seconds")

                ang = np.rad2deg(SO2.Log(pose[:2, :2]))
                shift = np.array([pose[0, 2], pose[1, 2]], dtype=np.float32)
                # print(ang, shift)
                # print(f"[DIRECT] Pose vector: {SE3.Log(pose).ravel()}")
                # dir_se2.append(np.array([ang, shift[0], shift[1]], dtype=np.float32))
                # ang, shift = convert_se3_estimate_to_planar_estimates(pose)
                # dir_se3.append(
                    
                # )
                # dir_se3.append(np.block([[SO3.Log(pose[:3, :3]).ravel()], [pose[:3, 3]]]).ravel())
                dir_se3.append(np.array([0, 0, SO2.Log(pose[:2, :2]), shift[0], shift[1], 0], dtype=np.float32))
                # DirectRegistration.visualize_registration(prev_image.copy(), image.copy(), ang, gamma * shift, "Direct Registration Result")
            except Exception as e:
                print(f"Direct Registration failed for {polar_file.name}: {e}")
                dir_se3.append(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32))


            # ### RADON REGISTRATION
            try:
                time_start = time.time()
                pose = rad_reg.register_images(prev_image, image, None, None,  None )
                time_end = time.time()
                print(f"Radon Registration took {time_end - time_start:.2f} seconds")

                ang = np.rad2deg(SO2.Log(pose[:2, :2]))
                shift = np.array([pose[0, 2], pose[1, 2]], dtype=np.float32)
                # print(ang, shift)
                # print(f"[DIRECT] Pose vector: {SE3.Log(pose).ravel()}")
                # dir_se2.append(np.array([ang, shift[0], shift[1]], dtype=np.float32))
                # ang, shift = convert_se3_estimate_to_planar_estimates(pose)
                # dir_se3.append(
                    
                # )
                # dir_se3.append(np.block([[SO3.Log(pose[:3, :3]).ravel()], [pose[:3, 3]]]).ravel())
                rad_se3.append(np.array([0, 0, SO2.Log(pose[:2, :2]), shift[0], shift[1], 0], dtype=np.float32))
                # RadonRegistration.visualize_registration(prev_image.copy(), image.copy(), ang, shift, "Radon Registration Result")
            except Exception as e:
                print(f"Radon Registration failed for {polar_file.name}: {e}")
                rad_se3.append(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32))



            
            # print("Ground truth relative pose (planar):", gt_rel_pose_planar)

            # Ground truth alignment
            # icp_reg.visualize_registration(prev_image, image, ang_gt, shift_gt * gamma) # Shift negative bc of opencv convention

            prev_image = image
            prev_image_polar = polar_img
            prev_image_name = Path(polar_file).name
            prev_pose = T_as
            
        else:
            prev_image = image
            prev_image_polar = polar_img
            prev_image_name = Path(polar_file).name
            prev_pose = T_as
                   
                # im_median = cv2.medianBlur(image, 5)
                # # im_aniso = cv2.ximgproc.anisotropicDiffusion(image, alpha=0.1, K=0.02, niters=1)
                # im_bilateral = cv2.bilateralFilter(image, 9, 75, 75)
                # cv2.imshow('image', image)
                # cv2.imshow('median', im_median)
                # # cv2.imshow('aniso', im_aniso)
                # cv2.imshow('bilateral', im_bilateral)
                # cv2.waitKey(0)

    n_img += 1


output_path = Path(dataset)
# Save results
np.savez(output_path.joinpath("icp_results.npz"), icp_se3=np.array(icp_se3, dtype=np.float32))
np.savez(output_path.joinpath("fourier_results.npz"), fou_se3=np.array(fou_se3, dtype=np.float32))
np.savez(output_path.joinpath("gt_results.npz"), gt_se3=np.array(gt_se3, dtype=np.float32))
np.savez(output_path.joinpath("init_results.npz"), init_se3=np.array(initial_se3, dtype=np.float32))
np.savez(output_path.joinpath("direct_results.npz"), dir_se3=np.array(dir_se3, dtype=np.float32))
np.savez(output_path.joinpath("radon_results.npz"), rad_se3=np.array(rad_se3, dtype=np.float32))
np.savez(output_path.joinpath("teaser_results.npz"), teas_se3=np.array(teas_se3, dtype=np.float32))
# # Visualize results


# ax = fig.add_subplot(1, 5, 5)
# ax.imshow(im_ref, interpolation="nearest", cmap=plt.cm.gray)
# ax.set_title("Original", fontsize=10)
# ax.set_xticks([])
# ax.set_yticks([])
# fig.tight_layout()
# plt.show()

# sigmaest = estimate_sigma(im_ref, average_sigmas=True)
# im_visushrink = denoise_wavelet(im_ref, method='BayesShrink', mode='soft', sigma=sigmaest / 4, rescale_sigma=True)
# im_visushrink4 = cv2.normalize(im_visushrink, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# # Concatenate images horizontally
# im_combined = np.hstack((im_ref, im_visushrink4))

# # Display the concatenated image
# cv2.imshow('Original and VisuShrink', im_combined)

# cv2.waitKey(1)

# # Find the keypoints and descriptors with AKAZE
# kp1, des1 = akaze.detectAndCompute(im_ref, None)
# kp2, des2 = akaze.detectAndCompute(im_1, None)

# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# # Apply ratio test
# good = []
# for m, n in matches:
#     if m.distance < 0.5 * n.distance:
#         good.append([m])

# # cv2.drawMatchesKnn expects list of lists as matches.
# im_matches = cv2.drawMatchesKnn(im_ref, kp1, im_1, kp2, good, None, flags=2)
# cv2.imshow('AKAZE matches', im_matches)
# cv2.waitKey(0)
