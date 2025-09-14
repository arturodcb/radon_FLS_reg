import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pywt
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.filters import unsharp_mask
from skimage.registration import phase_cross_correlation
from pathlib import Path
import time

from utils import convert_se3_estimate_to_planar_estimates
from image_reg import (
    ICPRegistration,
    FourierRegistration,
    DirectRegistration,
    SonarImageRegistration,
    RadonRegistration,
    RANSACRegistration,
    TEASERRegistration,
    FourierRotRegistration
)

import ciso8601
import copy

from uvnav_py.lib import FLS

from utils import to_cartesian, load_gt, readAgisoftTrajectory, is_mostly_black, create_fan_mask, plot_polar
from uvnav_py.utils import state_interp


dataset = "/media/arturo/Voyis/Quarry0211/truck_lm/"

gt = dataset + "cameras.xml"

polar_images = dataset + "sonar/"
data_path = Path(polar_images)
gt_path = Path(gt)

# Load states
import yaml
import pandas as pd
from pymlg import SO2, SO3, SE3, SE2
dataset = "quarry"
with open(Path.cwd().joinpath("config/" + dataset + ".yaml")) as f:
    dataset_params: dict[str, dict] = yaml.load(f, Loader=yaml.FullLoader)

discovery_base = dataset_params.pop("discovery_base")
blueprint_base = dataset_params.pop("blueprint_base")
sprintnav_base = dataset_params.pop("sprintnav_base")
sprintnavmini = dataset_params.pop("sprintnavmini")


data_path = Path(dataset_params.pop("data_path"))
output_path = Path(dataset_params.pop("output_path"))
sensor = "sonar"

prev_image = None
prev_image_polar = None
prev_image_name = None

icp_reg = ICPRegistration(
    r_max=10, r_min=0, azimuth_fov=130, elevation_fov=20, real_data=True
)
fourier_reg = FourierRegistration(r_max=10, r_min=0, azimuth_fov=130, elevation_fov=20)
dir_reg = DirectRegistration(r_max=10, r_min=0, azimuth_fov=130, elevation_fov=20)
radon_reg = RadonRegistration(
    r_max=10, r_min=0, azimuth_fov=130, elevation_fov=20
)
ransac_reg = RANSACRegistration(
    r_max=10, r_min=0, azimuth_fov=130, elevation_fov=20
)
teaser_reg = TEASERRegistration(
    r_max=10, r_min=0, azimuth_fov=130, elevation_fov=20, real_data=True
)
fourier_rot_reg = FourierRotRegistration(r_max=10, r_min=0, azimuth_fov=130, elevation_fov=20)

rel_posesICP = []
rel_posesFou = []
rel_posesDir = []
rel_posesRad = []
rel_posesTEAS = []

timesICP = []
timesFou = []
timesDir = []
timesRad = []
timesTEAS = []
# nanPose = SE2.identity()
nanPose = np.nan * np.eye(3)
nanPosese3 = np.nan * np.eye(4)
for trial, dataset_param in dataset_params.items():
    if (
        trial == "truck_lm"
    ):  # truck_lm start at frame 200 is good # conveyor_lm start at frame 200 is yaw excited
        sonar_dir = output_path.joinpath(trial, "csv", sensor + ".csv")

        data = pd.read_csv(sonar_dir, delimiter=",", index_col=False)
        # CHange directory to the one with sonar images
        data.loc[:, "image_file"] = data.loc[:, "image_file"].apply(
            lambda x: str(data_path.joinpath(trial, "sonar", x))
        )

        for i, img_file in enumerate(data.loc[:, "image_file"]):
            img_file = img_file.replace(
                "/media/arturo/El compa/", "/media/arturo/Voyis/"
            )

            if i % 5 == 0 and i > 480 and i < 715: # 480 truck lm for moment it start yawing 
                polar_file = (
                    Path(img_file)
                    .parents[1]
                    .joinpath("polar_images", Path(img_file).name)
                )

                polar_img = cv2.imread(str(polar_file), cv2.IMREAD_GRAYSCALE)

                image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

                ts = polar_file.stem.split("_")[
                    -2
                ]  # Assuming the timestamp is in the filename

                if is_mostly_black(image):
                    print(f"Skipping image {img_file} due to low pixel values.")
                    prev_image = None
                    continue

                # convert datetime to epoch time
                ts_converted = ciso8601.parse_datetime(ts.replace("-", "")).timestamp()

                # plot_polar(np.fliplr(polar_img), 130, 0, 10, None, title=Path(polar_file).name, real_data=True)

                # cv2.imshow("Image", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # user_in = input("Press Enter to continue, 's' to save: ")
                # if user_in == 's':
                #     plot_polar(np.fliplr(polar_img), 130, 0, 10, None, title=Path(polar_file).name, real_data=True)

                # cv2.imshow("image", image)
                # cv2.waitKey(50)
                # image = lee_filtered_uint8


                # fan_mask = create_fan_mask(image.shape, 25)
                # # rad = cv2.cvtColor(radon(image), cv2.COLOR_GRAY2BGR)
                # image_with_white_bg = image.copy()
                # image_with_white_bg[fan_mask == 0] = 255  #
                # # image_with_white_bg = image.copy()
                # # image_with_white_bg[~fan_mask] = 255
                # cv2.imshow("image with white bg", image_with_white_bg)
                # cv2.waitKey(0)
                if prev_image is not None:

                    print("Processing image: ", Path(img_file).name)
                    print("Reference image: ", prev_image_name)

                    gamma = image.shape[1] / (
                        2.0
                        * icp_reg.r_max
                        * np.sin(np.deg2rad(icp_reg.azimuth_fov) / 2.0)
                    )

                    ### ICP
                    # try:

                    #     pose = icp_reg.register_images(
                    #         prev_image, image, prev_image_polar, polar_img, None
                    #     )

                    #     ang, shift = convert_se3_estimate_to_planar_estimates(pose)

                    #     # SonarImageRegistration.visualize_registration(prev_image, image, ang, gamma*shift, "ICP Registration Result")

                    #     # icp_se2.append(np.array([ang, shift[0], shift[1]], dtype=np.float32))
                    #     rel_posesICP.append(pose)
                    #     timesICP.append(ts_converted)
                    # except Exception as e:
                    #     print(f"ICP Registration failed for {polar_file.name}: {e}")
                    #     rel_posesICP.append(nanPose)

                    ## FOURIER REGISTRATION
                    # try:
                    #     start_time = time.time()
                    #     pose = fourier_reg.register_images(
                    #         prev_image, image, prev_image_polar, polar_img, None
                    #     )
                    #     print(f"Fourier Registration took {time.time() - start_time:.2f} seconds")
                    #     ang = np.rad2deg(SO2.Log(pose[:2, :2]))
                    #     shift = np.array([pose[0, 2], pose[1, 2]], dtype=np.float32)
                    #     # ang, shift = convert_se3_estimate_to_planar_estimates(pose)
                    #     # SonarImageRegistration.visualize_registration(prev_image, image, ang, shift, "Fourier Registration Result")
                    #     # fou_se2.append(np.array([ang, shift[0], shift[1]], dtype=np.float32))

                    #     rel_posesFou.append(pose)
                    #     timesFou.append(ts_converted)

                    # except Exception as e:
                    #     print(f"Fourier Registration failed for {polar_file.name}: {e}")
                    #     rel_posesFou.append(nanPose)

                    # DIRECT REGISTRATION
                    # try:

                    #     pose = dir_reg.register_images(
                    #         prev_image, image, None, None, None
                    #     )

                    #     ang = np.rad2deg(SO2.Log(pose[:2, :2]))
                    #     shift = np.array([pose[0, 2], pose[1, 2]], dtype=np.float32)
                    #     # print(ang, shift)
                    #     # print(f"[DIRECT] Pose vector: {SE3.Log(pose).ravel()}")
                    #     # dir_se2.append(np.array([ang, shift[0], shift[1]], dtype=np.float32))
                    #     # ang, shift = convert_se3_estimate_to_planar_estimates(pose)
                    #     # dir_se3.append(

                    #     # )
                    #     # dir_se3.append(np.block([[SO3.Log(pose[:3, :3]).ravel()], [pose[:3, 3]]]).ravel())
                    #     # SonarImageRegistration.visualize_registration(prev_image.copy(), image.copy(), ang, gamma * shift, "Direct Registration Result")
                    #     rel_posesDir.append(pose)
                    #     timesDir.append(ts_converted)
                    # except Exception as e:
                    #     print(f"Direct Registration failed for {polar_file.name}: {e}")
                    #     rel_posesDir.append(nanPose)
                # Radon REGISTRATION
                    try:

                        t_start = time.time()
                        pose = radon_reg.register_images(
                            prev_image, image, prev_image_polar, polar_img, None
                        )
                        print(f"Radon Registration took {time.time() - t_start:.2f} seconds")

                        ang = np.rad2deg(SO2.Log(pose[:2, :2]))
                        shift = np.array([pose[0, 2], pose[1, 2]], dtype=np.float32)
                        # ang, shift = convert_se3_estimate_to_planar_estimates(pose)
                        # SonarImageRegistration.visualize_registration(prev_image, image, ang, shift, "Radon Registration Result")
                        # fou_se2.append(np.array([ang, shift[0], shift[1]], dtype=np.float32))
                        print(f"[RADON] Translation meters: {shift / gamma}")
                        rel_posesRad.append(pose)
                        timesRad.append(ts_converted)

                    except Exception as e:
                        print(f"Radon Registration failed for {polar_file.name}: {e}")
                        rel_posesRad.append(nanPose)

                    ### TEASER REGISTRATION
                    # try:
                    #     t_start = time.time()
                    #     pose = teaser_reg.register_images(
                    #         prev_image, image, prev_image_polar, polar_img, None
                    #     )
                    #     ang, shift = convert_se3_estimate_to_planar_estimates(pose)

                    #     # SonarImageRegistration.visualize_registration(prev_image, image, ang, gamma*shift, "TEASER Registration Result")

                    #     # rel_posesRad.append(pose)
                    #     # timesRad.append(ts_converted)
                    #     rel_posesTEAS.append(pose)
                    #     timesTEAS.append(ts_converted)
                    # # except Exception as e:
                    # #     print(f"ICP Registration failed for {polar_file.name}: {e}")
                    # #     rel_posesDir.append(nanPose)
                    # except Exception as e:
                    #     print(f"TEASER Registration failed for {polar_file.name}: {e}")
                    #     rel_posesTEAS.append(nanPosese3)

                    # Fourier Rotation REGISTRATION
                    # try:

                    # t_start = time.time()
                    # pose = fourier_rot_reg.register_images(
                    #     prev_image, image, prev_image_polar, polar_img, None
                    # )
                    # print(f"Fourier Rotation Registration took {time.time() - t_start:.2f} seconds")

                    # ang = np.rad2deg(SO2.Log(pose[:2, :2]))
                    # shift = np.array([pose[0, 2], pose[1, 2]], dtype=np.float32)
                    # # ang, shift = convert_se3_estimate_to_planar_estimates(pose)
                    # # SonarImageRegistration.visualize_registration(prev_image, image, ang, shift, "Radon Registration Result")
                    # # fou_se2.append(np.array([ang, shift[0], shift[1]], dtype=np.float32))
                    # print(f"[RADON] Translation meters: {shift / gamma}")
                    # rel_posesRad.append(pose)
                    # timesRad.append(ts_converted)

                    # except Exception as e:
                    #     print(f"Fourier Rotation Registration failed for {polar_file.name}: {e}")
                        # rel_posesRad.append(nanPose)

                    prev_image = image
                    prev_image_polar = polar_img
                    prev_image_name = Path(img_file).name
                else:
                    prev_image = image
                    prev_image_polar = polar_img
                    prev_image_name = Path(img_file).name
                # im_median = cv2.medianBlur(image, 5)
                # # im_aniso = cv2.ximgproc.anisotropicDiffusion(image, alpha=0.1, K=0.02, niters=1)
                # im_bilateral = cv2.bilateralFilter(image, 9, 75, 75)
                # cv2.imshow('image', image)
                # cv2.imshow('median', im_median)
                # # cv2.imshow('aniso', im_aniso)
                # cv2.imshow('bilateral', im_bilateral)
                # cv2.waitKey(0)

        # np.savez(
        #     output_path.joinpath(trial, "icp_results.npz"),
        #     icp_se3=np.vstack(rel_posesICP),
        #     times=timesICP,
        # )
        # np.savez(
        #     output_path.joinpath(trial, "fourier_results.npz"),
        #     fou_se3=np.vstack(rel_posesFou),
        #     times=timesFou,
        # )
        # np.savez(
        #     output_path.joinpath(trial, "direct_results.npz"),
        #     dir_se3=np.vstack(rel_posesDir),
        #     times=timesDir,
        # )
        np.savez(
            output_path.joinpath(trial, "radon_results.npz"),
            rad_se3=np.vstack(rel_posesRad),
            times=timesRad,
        )
        # np.savez(
        #     output_path.joinpath(trial, "teaser_results.npz"),
        #     teas_se3=np.vstack(rel_posesTEAS),
        #     times=timesTEAS,
        # )

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
