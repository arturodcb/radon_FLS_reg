import gtsam.noiseModel
import numpy as np
from typing import List
import cv2
from heapq import nlargest
import open3d as o3d
import open3d.core as o3c
from utils import create_fan_mask, resize_image
from scipy.signal import convolve2d
from utils import getPixelValue, getPatternPixelValue, getGradient, convert_se3_estimate_to_planar_estimates



from pymlg.numpy import SE3, SO3, SE2, SO2


import FLS



from skimage.util import img_as_float

class SonarImageRegistration:
    def __init__(self, r_max, r_min, azimuth_fov, elevation_fov):

        self.r_max = r_max
        self.r_min = r_min
        self.azimuth_fov = azimuth_fov
        self.elevation_fov = elevation_fov
        

        self.curr_fts = None
        self.prev_fts = None

        self.prev_image = None
        self.track_count = None

        self.detector = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB, 0, 3, 0.001, 4, 4, cv2.KAZE_DIFF_PM_G1)

        self.real_data = True
        self.k_strongest = 7  # Number of strongest pixels to consider
        self.z_threshold = 100  # Threshold for strong pixels
        if self.real_data:
            self.bearings = np.array([-65  , -64.05, -63.13, -62.24, -61.38, -60.54, -59.72, -58.93,
                                -58.15, -57.38, -56.63, -55.9 , -55.18, -54.47, -53.78, -53.09,
                                -52.42, -51.76, -51.11, -50.46, -49.83, -49.2 , -48.58, -47.97,
                                -47.36, -46.76, -46.17, -45.59, -45.01, -44.44, -43.87, -43.31,
                                -42.75, -42.2 , -41.65, -41.11, -40.57, -40.03, -39.5 , -38.98,
                                -38.46, -37.94, -37.42, -36.91, -36.4 , -35.9 , -35.4 , -34.9 ,
                                -34.4 , -33.91, -33.42, -32.94, -32.45, -31.97, -31.49, -31.02,
                                -30.54, -30.07, -29.6 , -29.13, -28.67, -28.2 , -27.74, -27.28,
                                -26.83, -26.37, -25.92, -25.47, -25.02, -24.57, -24.12, -23.67,
                                -23.23, -22.79, -22.35, -21.91, -21.47, -21.03, -20.6 , -20.16,
                                -19.73, -19.3 , -18.87, -18.44, -18.01, -17.58, -17.15, -16.73,
                                -16.3 , -15.88, -15.45, -15.03, -14.61, -14.19, -13.77, -13.35,
                                -12.93, -12.52, -12.1 , -11.68, -11.27, -10.85, -10.44, -10.02,
                                    -9.61,  -9.2 ,  -8.79,  -8.37,  -7.96,  -7.55,  -7.14,  -6.73,
                                    -6.32,  -5.91,  -5.5 ,  -5.09,  -4.68,  -4.28,  -3.87,  -3.46,
                                    -3.05,  -2.64,  -2.24,  -1.83,  -1.42,  -1.01,  -0.61,  -0.2 ,
                                    0.2 ,   0.61,   1.01,   1.42,   1.83,   2.24,   2.64,   3.05,
                                    3.46,   3.87,   4.28,   4.68,   5.09,   5.5 ,   5.91,   6.32,
                                    6.73,   7.14,   7.55,   7.96,   8.37,   8.79,   9.2 ,   9.61,
                                    10.02,  10.44,  10.85,  11.27,  11.68,  12.1 ,  12.52,  12.93,
                                    13.35,  13.77,  14.19,  14.61,  15.03,  15.45,  15.88,  16.3 ,
                                    16.73,  17.15,  17.58,  18.01,  18.44,  18.87,  19.3 ,  19.73,
                                    20.16,  20.6 ,  21.03,  21.47,  21.91,  22.35,  22.79,  23.23,
                                    23.67,  24.12,  24.57,  25.02,  25.47,  25.92,  26.37,  26.83,
                                    27.28,  27.74,  28.2 ,  28.67,  29.13,  29.6 ,  30.07,  30.54,
                                    31.02,  31.49,  31.97,  32.45,  32.94,  33.42,  33.91,  34.4 ,
                                    34.9 ,  35.4 ,  35.9 ,  36.4 ,  36.91,  37.42,  37.94,  38.46,
                                    38.98,  39.5 ,  40.03,  40.57,  41.11,  41.65,  42.2 ,  42.75,
                                    43.31,  43.87,  44.44,  45.01,  45.59,  46.17,  46.76,  47.36,
                                    47.97,  48.58,  49.2 ,  49.83,  50.46,  51.11,  51.76,  52.42,
                                    53.09,  53.78,  54.47,  55.18,  55.9 ,  56.63,  57.38,  58.15,
                                    58.93,  59.72,  60.54,  61.38,  62.24,  63.13,  64.05,  65 ]) 
        else:
            # TODO fix 520 hardcode
            self.bearings = np.linspace(-self.azimuth_fov/2, self.azimuth_fov/2, 520, dtype=np.float32)  # Bearings from -65 to 65 degrees
        # Additional initialization for ICP-specific parameters can be added here

    def register_images(self, ref_image: np.ndarray, curr_image: np.ndarray, ref_polar: np.ndarray = None, curr_polar: np.ndarray = None, aTb : np.ndarray = None)-> np.ndarray:
        # Placeholder for image registration logic
        # This should include steps like feature detection, matching, and transformation estimation
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def visualize_registration(ref_img, curr_img, angle_shift, translation_shift, name="Registration Result"):
        # Rotate the current image by the angle shift
        rot = cv2.getRotationMatrix2D(
                        (curr_img.shape[1] / 2, curr_img.shape[0]), -angle_shift, 1
                    )
        rotated_image = cv2.warpAffine(
            curr_img, rot, (curr_img.shape[1], curr_img.shape[0])
        )
        # Apply translation
        # Since translation is in pixels but with x forward and y right, need to match the OpenCV convention (x right and y down)
        t = np.float32([[1, 0, translation_shift[1]], [0, 1, -translation_shift[0]]])
        dst = cv2.warpAffine(
            rotated_image, t, (curr_img.shape[1], curr_img.shape[0])
        )

        black = np.zeros(curr_img.shape, dtype=np.uint8)
        ref_targ = resize_image(
            np.hstack((
                (cv2.merge([black, ref_img, black])),
                cv2.merge([black, black, curr_img]),
            )),
            0.5,
        )
        warp = resize_image(
            np.hstack((
                (
                    cv2.absdiff(
                        cv2.merge([black, black, curr_img]),
                        cv2.merge([black, ref_img, black]),
                    )
                ),
                cv2.absdiff(
                    cv2.merge([black, black, dst]),
                    cv2.merge([black, ref_img, black]),
                )),
            ),
            0.5,
        )


        cv2.imshow(
            name,
            np.vstack((ref_targ, warp)),
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()


   
class ICPRegistration(SonarImageRegistration):

    def __init__(self, r_max, r_min, azimuth_fov, elevation_fov, real_data: bool = False):
        super().__init__(r_max, r_min, azimuth_fov, elevation_fov)

        self.k_strongest = 10  # Number of strongest pixels to consider
        self.z_threshold = 70  # Threshold for strong pixels
        

        self.real_data = real_data

        if real_data:
            self.bearings = np.array([-65  , -64.05, -63.13, -62.24, -61.38, -60.54, -59.72, -58.93,
                                -58.15, -57.38, -56.63, -55.9 , -55.18, -54.47, -53.78, -53.09,
                                -52.42, -51.76, -51.11, -50.46, -49.83, -49.2 , -48.58, -47.97,
                                -47.36, -46.76, -46.17, -45.59, -45.01, -44.44, -43.87, -43.31,
                                -42.75, -42.2 , -41.65, -41.11, -40.57, -40.03, -39.5 , -38.98,
                                -38.46, -37.94, -37.42, -36.91, -36.4 , -35.9 , -35.4 , -34.9 ,
                                -34.4 , -33.91, -33.42, -32.94, -32.45, -31.97, -31.49, -31.02,
                                -30.54, -30.07, -29.6 , -29.13, -28.67, -28.2 , -27.74, -27.28,
                                -26.83, -26.37, -25.92, -25.47, -25.02, -24.57, -24.12, -23.67,
                                -23.23, -22.79, -22.35, -21.91, -21.47, -21.03, -20.6 , -20.16,
                                -19.73, -19.3 , -18.87, -18.44, -18.01, -17.58, -17.15, -16.73,
                                -16.3 , -15.88, -15.45, -15.03, -14.61, -14.19, -13.77, -13.35,
                                -12.93, -12.52, -12.1 , -11.68, -11.27, -10.85, -10.44, -10.02,
                                    -9.61,  -9.2 ,  -8.79,  -8.37,  -7.96,  -7.55,  -7.14,  -6.73,
                                    -6.32,  -5.91,  -5.5 ,  -5.09,  -4.68,  -4.28,  -3.87,  -3.46,
                                    -3.05,  -2.64,  -2.24,  -1.83,  -1.42,  -1.01,  -0.61,  -0.2 ,
                                    0.2 ,   0.61,   1.01,   1.42,   1.83,   2.24,   2.64,   3.05,
                                    3.46,   3.87,   4.28,   4.68,   5.09,   5.5 ,   5.91,   6.32,
                                    6.73,   7.14,   7.55,   7.96,   8.37,   8.79,   9.2 ,   9.61,
                                    10.02,  10.44,  10.85,  11.27,  11.68,  12.1 ,  12.52,  12.93,
                                    13.35,  13.77,  14.19,  14.61,  15.03,  15.45,  15.88,  16.3 ,
                                    16.73,  17.15,  17.58,  18.01,  18.44,  18.87,  19.3 ,  19.73,
                                    20.16,  20.6 ,  21.03,  21.47,  21.91,  22.35,  22.79,  23.23,
                                    23.67,  24.12,  24.57,  25.02,  25.47,  25.92,  26.37,  26.83,
                                    27.28,  27.74,  28.2 ,  28.67,  29.13,  29.6 ,  30.07,  30.54,
                                    31.02,  31.49,  31.97,  32.45,  32.94,  33.42,  33.91,  34.4 ,
                                    34.9 ,  35.4 ,  35.9 ,  36.4 ,  36.91,  37.42,  37.94,  38.46,
                                    38.98,  39.5 ,  40.03,  40.57,  41.11,  41.65,  42.2 ,  42.75,
                                    43.31,  43.87,  44.44,  45.01,  45.59,  46.17,  46.76,  47.36,
                                    47.97,  48.58,  49.2 ,  49.83,  50.46,  51.11,  51.76,  52.42,
                                    53.09,  53.78,  54.47,  55.18,  55.9 ,  56.63,  57.38,  58.15,
                                    58.93,  59.72,  60.54,  61.38,  62.24,  63.13,  64.05,  65 ]) 
        else:
            # TODO fix 520 hardcode
            self.bearings = np.linspace(-self.azimuth_fov/2, self.azimuth_fov/2, 520, dtype=np.float32)  # Bearings from -65 to 65 degrees
        # Additional initialization for ICP-specific parameters can be added here

    def register_images(self, ref_image: np.ndarray, curr_image: np.ndarray, ref_polar: np.ndarray = None, curr_polar: np.ndarray = None, aTb : np.ndarray = None):
        # Placeholder for ICP registration logic
        # self.ref_img = ref_image
        # self.curr_img = curr_image

        # ref_image = cv2.fastNlMeansDenoising(ref_image, None, 10, 7, 21)
        # curr_image = cv2.fastNlMeansDenoising(curr_image, None, 10, 7, 21)
        # ref_polar = cv2.fastNlMeansDenoising(ref_polar, None, 10, 7, 21)
        # curr_polar = cv2.fastNlMeansDenoising(curr_polar, None, 10, 7, 21)
        # This should include steps like point cloud generation, nearest neighbor search, and transformation estimation
        pcd = self.extract_features(curr_image, curr_polar)
        ref_pcd = self.extract_features(ref_image, ref_polar)


        if pcd.point.positions.shape[0]<2 or ref_pcd.point.positions.shape[0] < 2:
            print("[ICP] Not enough features to register images.")
            return np.eye(4, dtype=np.float32)

        if aTb is None:
            trans_init = np.eye(4, dtype=np.float64)
        else:
            trans_init = aTb
        threshold = 0.5  # 10 cm threshold for ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
                                pcd.to_legacy(), ref_pcd.to_legacy(), threshold, trans_init,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        
        # Return the transformation matrix, Affine transformation matrix
        # TODO This transfomration must be in pixels in translation
        pose = reg_p2p.transformation  # barT_ij


        angle_shift, translation_shift = convert_se3_estimate_to_planar_estimates(pose)

        print(f"[ICP] Angle (degrees): {angle_shift}")
        print(f"[ICP] Shift (meters): {translation_shift}")
        
        return pose
                    
    def extract_features(self, image: np.ndarray, polar_image: np.ndarray):
        """
        Return the k strongest pixels along each azimuth
        that are above a certain z_threshold.

        Args:
            image (numpy.ndarray): The input image.
            k (int): The number of strongest pixels to return.
            z_threshold (float): The threshold above which pixels are considered strong.

        Returns:
            numpy.ndarray: Indices of the k strongest pixels.
        """
        # Flatten the image and get the indices of the k largest values
        if not self.real_data:
            polar_image = np.flipud(polar_image)

        FLSmodel = FLS.Model(r_min = self.r_min, r_max = self.r_max, azimuth_fov = self.azimuth_fov, elevation_fov = self.elevation_fov,
           width_image = image.shape[1])

        rbin = (self.r_max - self.r_min) / polar_image.shape[0]
        kps = []
        kps_cart = []
        xyz_pc = []
        for i in range(polar_image.shape[1]):
            column = polar_image[:, i]
            # Filter out values below the threshold, set them to 0
            # column = np.where(column < z_threshold, 0, column)
            nlarg_idx = nlargest(self.k_strongest, range(len(column)), key=column.__getitem__)
            # Filter out indices below the z_threshold
            for idx in nlarg_idx:
                if column[idx] >= self.z_threshold:
                    kps.append((idx, i))
        
                    # Transform the polar pixel coordinates to Cartesian coordinates
                    # First, pixel to range and bearing

                    # assuming equally spaced range_bins
                    r = (polar_image.shape[0] - idx)*rbin + self.r_min  # range
                    # bearings from -65 to 65 degrees, assuming equally spaced bearing_bins
                    # for now but spacing is not linear
                    b = self.bearings[i]  # bearing
                    
                    # Now, convert range and bearing to Cartesian coordinates
                    #Assuming bearing is in degrees and range is in meters
                    x = r * np.cos(np.deg2rad(b))
                    y = r * np.sin(np.deg2rad(b))
                    xyz_pc.append(np.array([x, y, 0]))  # z is 0 for sonar
                    #Now convert to pixel coordinates
                    kps_cart.append(FLSmodel.project(np.array([x, y, 0])))

        
        pcd = o3d.t.geometry.PointCloud(np.vstack(xyz_pc, dtype=np.float64))
        # pcd = pcd.to(o3c.Device("cuda:0"))

        # Show keypoints in the image
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for visualization
        # polar_image = cv2.cvtColor(polar_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for visualization
        # for idx, i in kps:
        #     cv2.circle(polar_image, (i, idx), 3, (0,
        #                255, 0), -1)  # Draw a circle at the keypoint location
        # for pt in kps_cart:
        #     cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)  # Draw a circle at the keypoint location
        # cv2.imshow("Keypoints", polar_image)
        # cv2.imshow("Keypoints Cartesian", image)
        # cv2.waitKey(0)
        return pcd 

class FourierRegistration(SonarImageRegistration):
    def __init__(self, r_max, r_min, azimuth_fov, elevation_fov):
        super().__init__(r_max, r_min, azimuth_fov, elevation_fov)
        
        # Additional initialization for Fourier-specific parameters can be added here
        self.blurred_mask = None

    def register_images(self, ref_image: np.ndarray, curr_image: np.ndarray, ref_polar: np.ndarray = None, curr_polar: np.ndarray = None, aTb : np.ndarray = None):
        # Placeholder for Fourier registration logic
        # This should include steps like Fourier transform, phase correlation, and transformation estimation
        # ref_image = cv2.fastNlMeansDenoising(ref_image, None, 10, 7, 21)
        # curr_image = cv2.fastNlMeansDenoising(curr_image, None, 10, 7, 21)
        # ref_polar = cv2.fastNlMeansDenoising(ref_polar, None, 10, 7, 21)
        # curr_polar = cv2.fastNlMeansDenoising(curr_polar, None, 10, 7, 21)

        if aTb is not None:
            init_angle_shift = np.rad2deg(SO2.Log(aTb[:2, :2]))
            init_shift = np.array([aTb[0, 2], aTb[1, 2]], dtype=np.float32)


            polar_shift_pix = init_angle_shift * curr_polar.shape[1] / 130.0  # 130 degrees is azimuth fov of sonar
            # First, shift polar image by the angle shift
            t = np.float32([[1, 0, polar_shift_pix], [0, 1, 0]])
            curr_polar_init = cv2.warpAffine(
                curr_polar, t, (curr_polar.shape[1], curr_polar.shape[0])
            )
        else:
            curr_polar_init = curr_polar
            init_angle_shift = 0

        shift = FourierRegistration.fourier_registration(
            ref_polar, curr_polar_init, polar=True
        )
        print(f"[Fourier] Delta POLAR shift (y, x): {shift}")
        #print(f"Detected POLAR subpixel offset (, x): {shift}")
        angle_shift = (
            shift[1] * self.azimuth_fov / curr_polar.shape[1]
        )  # 130 degrees is azimuth fov of sonar. This is a linear
        # approximation of the angle shift
        print(f"[Fourier] Delta angle (degrees): {angle_shift}")
        
        # Now rotate the image by the angle shift + init_angle_shift (they are in different frames so init is negative)
        M = cv2.getRotationMatrix2D(
            (curr_image.shape[1] / 2, curr_image.shape[0]), -angle_shift - init_angle_shift, 1
        )
        rotated_image = cv2.warpAffine(
            curr_image, M, (curr_image.shape[1], curr_image.shape[0])
        )
        
        if self.blurred_mask is None:

            fan_mask = create_fan_mask(curr_image.shape, 25)
            blurred_mask = cv2.blur(fan_mask, (19, 19))

        rotated_mask = cv2.warpAffine(
            blurred_mask, M, (curr_image.shape[1], curr_image.shape[0])
        )
        # cv2.imshow('rotated image', np.array(rotated_image*rotated_mask, dtype=np.uint8))
        # cv2.imshow('rotated mask', np.array(prev_image*blurred_mask, dtype=np.uint8))
        # cv2.imshow('rotated mask')
        # shift, error, diffphase = phase_cross_correlation(prev_image, rotated_image, disambiguate=True)

        if aTb is not None:
            init_shift = np.array([aTb[0, 2], aTb[1, 2]], dtype=np.float32)
            gamma = curr_image.shape[1] / (
                    2.0 * self.r_max * np.sin(np.deg2rad(self.azimuth_fov) / 2.0)
                )
            init_shift = init_shift * gamma  # Convert shift from meters to pixels
            t = np.float32([[1, 0, init_shift[1]], [0, 1, -init_shift[0]]])
            rotated_image_init = cv2.warpAffine(
                rotated_image, t, (rotated_image.shape[1], rotated_image.shape[0])
            )
            rotated_mask_init = cv2.warpAffine(
                rotated_mask, t, (rotated_mask.shape[1], rotated_mask.shape[0])
            )
        else:
            rotated_image_init = rotated_image
            rotated_mask_init = rotated_mask

        shift = FourierRegistration.fourier_registration(
            ref_image,
            rotated_image_init,
            mask=blurred_mask,
            mask_2=rotated_mask_init,
        )
        # print(f"Detected subpixel CUSTOM offset (y, x): {shift_c}")
        print(f"[Fourier] Delta Shift (pixels): { np.array([-shift[0], shift[1]])}")

        # Return \theta_ab and shift following x forward y right convention (sonar convention)
        if aTb is not None:

            total_angle_shift = init_angle_shift + angle_shift
            total_shift = np.array([-shift[0], shift[1]], dtype=np.float32) + init_shift
        else:
            total_angle_shift = angle_shift
            total_shift = np.array([-shift[0], shift[1]], dtype=np.float32)

        print(f"[Fourier] Angle (degrees): {total_angle_shift}")
        print(f"[Fourier] Shift (pixels): {total_shift}")

        pose = SE2.from_components(
            SO2.Exp(np.deg2rad(total_angle_shift)),
            np.array([total_shift[0], total_shift[1]], dtype=np.float32)
        )

        return pose  # shift in pixels, x right, y down
    
    @staticmethod
    def fourier_registration(image1, image2, mask=None, mask_2 = None, polar=False):
        if mask is None:
            mask = np.ones(image1.shape)
            #zeros on the edges
            mask[0:3, :] = 0
            mask[-3:, :] = 0
            mask[:, 0:3] = 0
            mask[:, -3:] = 0

            mask = cv2.GaussianBlur(mask, (15, 15), 0)
        if mask_2 is None:
            mask_2 = np.ones(image2.shape)
            mask2 = np.ones(image2.shape)
            #zeros on the edges
            mask2[0:3, :] = 0
            mask2[-3:, :] = 0
            mask2[:, 0:3] = 0
            mask2[:, -3:] = 0

            mask2 = cv2.GaussianBlur(mask2, (15, 15), 0)
        image1 = image1 * mask
        image2 = image2 * mask_2
        
        #convert image from 8U to 32F
        ref_ = image1.astype(np.float32) 
        curr_ = image2.astype(np.float32) 

        # (y,x), response = cv2.phaseCorrelate(curr_, ref_)
        # print("OPENCV SHIFT (x, y):", (-x, y))

        # return (x, y)

        # M = cv2.getOptimalDFTSize(ref_.shape[0])
        # N = cv2.getOptimalDFTSize(ref_.shape[1])

        # if(M != ref_.shape[0] or N != ref_.shape[1]):
            
        #         padref = cv2.copyMakeBorder(ref_, 0, M - ref_.shape[0], 0, N - ref_.shape[1], cv2.BORDER_CONSTANT, 0)
        #         padcur = cv2.copyMakeBorder(curr_, 0, M - curr_.shape[0], 0, N - curr_.shape[1], cv2.BORDER_CONSTANT, 0)
               
        # else:   
        #         padref = ref_
        #         padcur = curr_
            
        # f1 = cv2.dft(padref, flags=cv2.DFT_REAL_OUTPUT)
        # f2 = cv2.dft(padcur, flags=cv2.DFT_REAL_OUTPUT)

        # Compute the 2D Fourier Transform of the images
        f1 = np.fft.rfft2(image1)
        f2 = np.fft.rfft2(image2)
        
        # Compute the cross-power spectrum
        cps = np.conj(f2) * f1 / np.abs(np.conj(f2) * f1)
        
        # Compute the cross-correlation function
        ccf = np.fft.irfft2(cps)

        # ccf = cv2.dft(cps, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE )

        # apply a low pass filter to the cross-correlation function
        fir_filter_2d = np.ones((15, 15)) / 15**2
        if polar:
            fir_filter_2d = np.ones((15, 9)) / (15*9)
        
        blurred_image_data = convolve2d(ccf, fir_filter_2d, mode='same', boundary='wrap')



        # plt.imshow(np.abs(blurred_image_data))
        # plt.show()
        
        # Compute the shift in the frequency domain
        y, x = np.unravel_index(np.argmax(ccf), ccf.shape)

        if y > ccf.shape[0] // 2:
            y -= ccf.shape[0]
        if x > ccf.shape[1] // 2:
            x -= ccf.shape[1]
        
        return (y, x)


from gtsam.utils.test_case import GtsamTestCase

import gtsam
from gtsam import CustomFactor, Pose2, Values, Rot2, Pose3
import FLS
from functools import partial

class DirectRegistration(SonarImageRegistration):
    def __init__(self, r_max, r_min, azimuth_fov, elevation_fov):
        super().__init__(r_max, r_min, azimuth_fov, elevation_fov)

    def register_images(self, ref_image: np.ndarray, curr_image: np.ndarray, ref_polar: np.ndarray = None, curr_polar: np.ndarray = None, aTb : np.ndarray = None):
        import pywt
        from skimage.util import img_as_float, img_as_ubyte


        # gauss_curr = cv2.pyrDown(cv2.pyrDown(curr_image))
        # gauss_ref = cv2.pyrDown(cv2.pyrDown(ref_image))

        # coeffs2 = pywt.dwt2(img_as_float(curr_image), 'bior1.3')
        # LL, (LH, HL, HH) = coeffs2
        # coeffs3 = pywt.dwt2(LL, 'bior1.3')
        # LL3, (LH, HL, HH) = coeffs3
        # coeffs4 = pywt.dwt2(LL3, 'bior1.3')
        # LL4, (LH, HL, HH) = coeffs4
        # wav_curr = cv2.normalize(LL, None, curr_image.min(), curr_image.max(), cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # # cv2.imshow("Wavelet and Gaussian", np.hstack([wavelet_down[0:min_dim, :], gauss_down3[0:min_dim, :]]))
        # # cv2.waitKey(0)

        # coeffs2 = pywt.dwt2(img_as_float(ref_image), 'bior1.3')
        # LL, (LH, HL, HH) = coeffs2
        # coeffs3 = pywt.dwt2(LL, 'bior1.3')
        # LL3, (LH, HL, HH) = coeffs3
        # coeffs4 = pywt.dwt2(LL3, 'bior1.3')
        # LL4, (LH, HL, HH) = coeffs4
        # wav_ref = cv2.normalize(LL4, None, ref_image.min(), ref_image.max(), cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # curr_image = gauss_curr
        # ref_image = gauss_ref
        
        # cv2.imshow("current", curr_image)
        # cv2.imshow("Wavelet Current", wav_curr)
        # cv2.imshow("Gauss", cv2.pyrDown(curr_image))
        # cv2.waitKey(0)

        # cv2.fastNlMeansDenoising(curr_image, curr_image, 10, 7, 21)
        feat_image = cv2.fastNlMeansDenoising(ref_image, None, 10, 7, 21)
        

        features = self.extract_features(feat_image)

        if len(features) < 2:
            print("[DIRECT] Not enough features to register images.")
            return np.eye(3, dtype=np.float32)

        pyr_levels = 4

        pyr_curr = self.form_pyramid(curr_image, pyr_levels)
        pyr_ref = self.form_pyramid(ref_image, pyr_levels)
        # # gradients on reference image
        # _, grad_x, grad_y = DirectRegistration.gradient_image(ref_image) 

        if aTb is None:
            aTb = np.eye(3, dtype=np.float32)

        aTb_check = aTb.copy()
        for i in range(pyr_levels-1 , -1, -1):
            # Scale measurements and fls model to the current pyramid level
            features_scaled = []
            for feat in features:
                FLSmodel = FLS.Model(
                    r_min=self.r_min , r_max=self.r_max , azimuth_fov=self.azimuth_fov, elevation_fov=self.elevation_fov,
                    width_image=pyr_curr[i].shape[1]
                )
                meas_scaled = FLS.Data(
                        model=FLSmodel,
                        uv=np.array([feat.value[0] / (2**i), feat.value[1] / (2**i)], dtype=np.uint32),
                        stamp=0.0
                    )
                
        
                features_scaled.append(meas_scaled)


            # Downscale the images
            aTb_check = self.align_image_pair(pyr_ref[i], pyr_curr[i], features_scaled, aTb_check)

        # print(f"Optimized transformation: {aTb_check}")
    
        #vector = SE3.Log(aTb_check.matrix()).ravel()
        # print(f"Optimized transformation: {aTb_check}")
        # print(f"Delta Pose: {SE3.Log(np.linalg.inv(aTb) @ aTb_check).ravel()}")
        

        ## SE3 setup
        # angle = np.rad2deg(SO3.Log(aTb_check[:3,:3])[2][0])  # angle in degrees
        # shift = np.array([aTb_check[0, 3], aTb_check[1, 3]], dtype=np.float32)  # translation in meters
        # print(f"[DIRECT] Angle (degrees): {angle}")
        # print(f"[DIRECT] Shift (meters): {shift}")

        # SO2 setup
        angle = np.rad2deg(SO2.Log(aTb_check[:2,:2]))  # angle in degrees
        shift = np.array([aTb_check[0, 2], aTb_check[1, 2]], dtype=np.float32)  # translation in meters
        print(f"[DIRECT] Angle (degrees): {angle}")
        print(f"[DIRECT] Shift (meters): {shift}")

        # Implement direct registration logic here
        return aTb_check  # Return angle in degrees and translation in meters (x forward, y right)


    def align_image_pair(self, ref_image: np.ndarray, curr_image: np.ndarray,  features : List[FLS.Data], aTb: np.ndarray = None):
        # DO a gradual alignment, where downcscaled images are aligned first 
        # and then use the estimated transformation to align a higher resolution image
        fg = gtsam.NonlinearFactorGraph()
        # if aTb is None:
        #     aTb = np.eye(4, dtype=np.float32)
        # initial_pose = aTb
        aTb_gtsam = Pose2(SO2.Log(aTb[:2, :2]), aTb[:2, 2])  # Initial guess for the transformation (no translation or rotation)
        # aTb_gtsam = Pose3(aTb)
        noise_model = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(10.0), gtsam.noiseModel.Isotropic.Sigma(1, 1) ) # Example noise model
        #noise_model = gtsam.noiseModel.Isotropic.Sigma(1, 1)  # Example noise model
        unknown = gtsam.symbol("x", 0)

        def error_func(feat: FLS.Data, this: CustomFactor, v: gtsam.Values,
                       H: List[np.ndarray]):
            """
            Error function that projects the features of a 
            :param this: reference to the current CustomFactor being evaluated
            :param v: Values object
            :param H: list of references to the Jacobian arrays
            :return: the non-linear error
            """
            key1 = this.keys()[0]
            T_ab = v.atPose2(key1).matrix()

            rb = feat.model.convert_to_bearing_range(feat.value) # x y pixel
            p_a = feat.model.to_cartesian(np.array([rb[0], 0, rb[1]])) # az el r -> x y z
            ## SE3 setup
            # p_a_hom = np.array([p_a[0], p_a[1], 0, 1])  # Convert to homogeneous coordinates in 2D (z is 0)
            # p_b_hom = SE3.inverse(T_ab) @ p_a_hom  # Transform the point from frame A to frame B
            # uv_b = feat.model.project(p_b_hom[:3])  # Project the point to the image plane

            ## SE2 setup
            p_a_hom = np.array([p_a[0], p_a[1], 1])  # Convert to homogeneous coordinates in 2D 
            p_b_hom = SE2.inverse(T_ab) @ p_a_hom
            uv_b = feat.model.project(np.array([p_b_hom[0], p_b_hom[1], 0]))  # Project the point to the image plane for SE2
                


            # TODO what to do if uv_a is out of bounds?
            # if uv_a[0] < 0 or uv_a[0] >= ref_image.shape[1] or uv_a[1] < 0 or uv_a[1] >= ref_image.shape[0]:
            #     if H is not None:
            #         H[0] = np.zeros((1, 6))  # If the point is out of bounds, return zero Jacobian

            #     return np.array([1e6])  # If the point is out of bounds, return large error that will be ignored

            try:
                intensity_curr =  getPatternPixelValue(curr_image, uv_b) # Get the intensity from the reference image
                intensity_ref = getPatternPixelValue(ref_image, feat.value)  # Get the intensity from the current image


                error = np.array([intensity_curr - intensity_ref])  # Compute the error as the difference in intensities
                # error = np.array([0, 0]).reshape((2, 1))  # Compute the error as the difference in pixel coordinates
                if H is not None:
                    # Jacobian of intensity with respect to the point
                    # D = np.array([[1, 0, 0], [0, 1, 0]])  # Identity matrix for the Jacobian of intensity for SE2
                    
                    # h= 1
                    # J_I_uv = np.array([ (getPixelValue(ref_image, uv_a + h*np.array([1, 0])) - getPixelValue(ref_image,uv_a - h*np.array([1, 0]))) / (2*h),
                    #                     (getPixelValue(ref_image, uv_a + h*np.array([0, 1])) - getPixelValue(ref_image, uv_a - h*np.array([0, 1]))) / (2*h)]).reshape((1, -1))
                    J_I_uv = getGradient(curr_image, uv_b)  # Jacobian of the intensity with respect to the pixel coordinates
                     # Jacobian of the projection with respect to the point in frame A
                     ## SE3 setup
                    # D = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # Identity matrix for the Jacobian of intensity
                    # J_uv_pb = feat.model.jacobian_projection(p_b_hom[:3])   # Jacobian of the projection with respect to the point in frame A
                    # J_pb_Tab = -D  @  SE3.odot(p_b_hom)  # Jacobian of the point with respect to the transformation
                    
                    ## SE2 setup
                    J_uv_pb = feat.model.jacobian_projection(np.array([p_b_hom[0], p_b_hom[1], 0]))
                    # In GTSAM, convention for xi is v, w, as opposed to w, v
                    X = np.zeros((3, 3))
                    X[0:2, 2] = SO2.odot(p_b_hom[0:2]).ravel()
                    X[0:2, 0:2] = np.identity(2)
                    J_pb_Tab = - X

                    H[0] =  J_I_uv @ J_uv_pb @ J_pb_Tab # J_uv_pa * J_pa_Tab is validated
            except Exception as e:
                error = np.array([1e6])  # If the point is out of bounds, return large error that will be ignored
                if H is not None:
                    # H[0] = np.zeros((1, 6))
                    H[0] = np.zeros((1, 3))  # If the point is out of bounds, return zero Jacobian

            # delta = np.array([getPixelValue(ref_image, uv_b + np.array([1, 0])) - getPixelValue(ref_image,uv_b - np.array([1, 0])) ,
            #                     getPixelValue(ref_image, uv_b + np.array([0, 1])) - getPixelValue(ref_image, uv_b - np.array([0, 1]))] )

            # if (np.linalg.norm(delta) < 10.0):
            #     return np.array([0])  #  Jacobian is small enough, return 0 error
            


            return error

        # def compute_jacobian_finite_diff(feat, T_ab):
        #     # Check that the feature is rightly projected
        #     rb = feat.model.convert_to_bearing_range(feat.value) # x y pixel
        #     p_b = feat.model.to_cartesian(np.array([rb[0], 0, rb[1]])) # az el r -> x y z
        #     p_b_hom = np.array([p_b[0], p_b[1], 0, 1])  # Convert to homogeneous coordinates in 3D (z is 0)


        #     p_a_hom = T_ab @ p_b_hom  # Transform the point from frame A to frame B

        #     uv_a = feat.model.project(p_a_hom[:3])  # Project the point to the image plane
        #     intensity_ref = getPixelValue(ref_image, uv_a)  # Get the intensity from the reference image

        #     dim = 6
        #     J = np.zeros((1, dim))  # Jacobian of the error with respect to the transformation parameters
        #     h= 1e-7
        #     for i in range(dim):
        #         dx = np.zeros(dim)
        #         dx[i] = h # Small perturbation

        #         T_ab_pert = T_ab @ SE3.Exp(dx)  # Perturb the transformation



        #         p_a_hom = T_ab_pert @ p_b_hom  # Transform the point from frame A to frame B

        #         # uv_a_pert = feat.model.project(np.array([p_a_hom[0], p_a_hom[1], 0]))  # Project the point to the image plane
        #         uv_a_pert = feat.model.project(p_a_hom[:3])  # Project the point to the image plane
        #         # uv_a_pert = feat.value + dx
        #         intensity_ref_dx = getPixelValue(ref_image, uv_a_pert)  # Get the intensity from the reference image
                


        #         error = intensity_ref_dx - intensity_ref
        #         # error = uv_a_pert - uv_a  # Compute the error as the difference in pixel coordinates
        #         J[: , i] = error / h  # Compute the finite difference approximation of the Jacobian

            # print(f"Jacobian for feature {feat.value}: {J}")
            # J_anal = np.array([ getPixelValue(grad_x, uv_a), getPixelValue(grad_y, uv_a) ])

            # print(f"Analytical Jacobian for feature {feat.value}: {J_anal}")
            # J_I_uv = np.array([ (getPixelValue(ref_image, uv_a + h*np.array([1, 0])) - getPixelValue(ref_image,uv_a - h*np.array([1, 0]))) / (2*h),
            #                         (getPixelValue(ref_image, uv_a + h*np.array([0, 1])) - getPixelValue(ref_image, uv_a - h*np.array([0, 1]))) / (2*h)]).reshape((1, -1))
            # print(f"Jacobian of the intensity for feature {feat.value}: {J_I_uv}")
            # return J

        # After feature extraction, align the image by solving the optimization problem
        # gtsam python bindings can be used for this

        # curr_feature_image = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)
        # J_list = []
        for feat in features:
            # Check that the feature is rightly projected
            # rb = feat.model.convert_to_bearing_range(feat.value) # x y pixel
            # p_a = feat.model.to_cartesian(np.array([rb[0], 0, rb[1]])) # az el r -> x y z
            # p_a_hom = np.array([p_a[0], p_a[1],  1])  # Convert to homogeneous coordinates in 3D (z is 0)
            # p_b_hom = SE2.inverse(aTb) @ p_a_hom  # Transform the point from frame A to frame B

            # uv_b = feat.model.project(np.array([p_b_hom[0], p_b_hom[1], 0]))  # Project the point to the image plane
            # # uv_b = feat.model.project(p_b_hom[:3])  # Project the point to the image plane
            # # J_list.append(compute_jacobian_finite_diff(feat, aTb_gtsam.matrix()))
            
            # cv2.circle(curr_feature_image, (int(uv_b[0]), int(uv_b[1])), 3, (0, 255, 0), -1)
        
            # cv2.circle(curr_feature_image, (int(feat.value[0]), int(feat.value[1])), 3, (255, 255, 255), -1)  # Draw a circle at the feature location

            cf = CustomFactor(noise_model, [unknown], partial( error_func, feat))
            fg.add(cf)


        
        # cv2.imshow("Projected Features", curr_feature_image)

        # cv2.waitKey(0)

        val = Values()
        val.insert(unknown, aTb_gtsam)  # Initial guess for the first pose
        
        # fg.print("Factor Graph:")
        # fg.error(val)
        gfg = fg.linearize(val)
        a, b = gfg.jacobian()
        # print("[DIRECT] Condition number ", np.linalg.cond(a.T @ a))

        # params = gtsam.LevenbergMarquardtParams()
        # params.setMaxIterations(1000)
        # optimizer = gtsam.LevenbergMarquardtOptimizer(fg, val, params)

        params = gtsam.GaussNewtonParams()
        params.setMaxIterations(1000)
        optimizer = gtsam.GaussNewtonOptimizer(fg, val, params)
        result = optimizer.optimize()

        aTb_check = result.atPose2(unknown)

        # print(f"[DIRECT] Optimized transformation: {aTb_check.matrix()}")

        return aTb_check.matrix()  # Return the optimized transformation matrix

    def extract_features(self, image: np.ndarray, polar_image: np.ndarray = None):
        """
        Extract features from the image using direct methods.
        This could involve edge detection, corner detection, etc.
        """
        window_name = ('Sobel Demo - Simple Edge Detector')


        fan_mask = create_fan_mask(image.shape, 25)
        blurred_mask = cv2.GaussianBlur(fan_mask, (55, 55), 20)

        # cv2.imshow("Fan mask", blurred_mask)
        # cv2.waitKey(0)

        masked_image = np.array(image * blurred_mask, dtype=np.uint8)

        # cv2.imshow("masked image", masked_image)
        
        # cv2.imshow('Original Image', image)
        # cv2.waitKey(0)

        grad_mag, grad_x, grad_y  = DirectRegistration.gradient_image(masked_image)
        
        

        FLSmodel = FLS.Model(r_min = self.r_min, r_max = self.r_max, azimuth_fov = self.azimuth_fov, elevation_fov = self.elevation_fov,
           width_image = image.shape[1])
        # cv2.imshow(window_name, grad_mag)
        # cv2.waitKey(0)
        
        # Grid the image and extract one feature per cell (the one with the highest gradient magnitude)
        # only those above a certain threshold are considered
        # Grid the image and extract one feature per cell
        grid_size = 50  # Size of each grid cell
        gradient_threshold = 80  # Minimum gradient magnitude threshold
        
        features = []
        
        # Calculate grid dimensions
        rows = image.shape[0] // grid_size
        cols = image.shape[1] // grid_size
        
        for i in range(rows):
            for j in range(cols):
                # Define cell boundaries
                y_start = i * grid_size
                y_end = min((i + 1) * grid_size, image.shape[0])
                x_start = j * grid_size
                x_end = min((j + 1) * grid_size, image.shape[1])
                
                # Extract the cell from the gradient image
                cell = grad_mag[y_start:y_end, x_start:x_end]
                
                # Find the maximum gradient magnitude in this cell
                max_val = np.max(cell)
                
                # Only consider if above threshold
                if max_val >= gradient_threshold:
                    # Find the position of maximum gradient in the cell
                    max_pos = np.unravel_index(np.argmax(cell), cell.shape)
                    
                    # Convert to image coordinates
                    global_y = y_start + max_pos[0]
                    global_x = x_start + max_pos[1]
                    
                    # Convert to FLS data format
                    meas = FLS.Data(
                        model=FLSmodel,
                        uv=np.array([global_x, global_y], dtype=np.uint32),
                        stamp=0.0
                    )

                    features.append(meas)
        
        # Visualize the features
        feature_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for meas in features:
            cv2.circle(feature_image, (meas.value[0], meas.value[1]), 3, (0, 255, 0), -1)
        
        # cv2.imshow("Grid Features", feature_image)
        # cv2.imshow(window_name, grad_mag)





        return features
    
    def form_pyramid(self, image: np.ndarray, levels: int = 3):
        """
        Create a Gaussian pyramid of the image.
        """
        pyramid = [image]
        for i in range(levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid
    

    
    @staticmethod
    def gradient_image(image: np.ndarray):
        """
        Compute the gradient of the image using Sobel operator.
        """
        # cv2.fastNlMeansDenoising(image, image, 7, 7, 21)  # Denoise the image
        guided_filter = cv2.ximgproc.createGuidedFilter(image, 15, 100)
        guided_filter.filter(image, image)

        # Gradient-X
        grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, 7)
        # Gradient-Y
        grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, 7)

        # cv2.imshow('Gradient X', cv2.convertScaleAbs(grad_x))
        # cv2.imshow('Gradient Y', cv2.convertScaleAbs(grad_y))
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0), grad_x, grad_y
    
from skimage.transform import radon

class RadonRegistration(SonarImageRegistration):
    def __init__(self, r_max, r_min, azimuth_fov, elevation_fov):
        super().__init__(r_max, r_min, azimuth_fov, elevation_fov)
        # Additional initialization for Fourier-specific parameters can be added here
        self.blurred_mask = None

    def register_images(self, ref_image: np.ndarray, curr_image: np.ndarray, ref_polar: np.ndarray = None, curr_polar: np.ndarray = None, aTb : np.ndarray = None):
        # Placeholder for Fourier registration logic
        # This should include steps like Fourier transform, phase correlation, and transformation estimation
        # ref_image = cv2.fastNlMeansDenoising(ref_image, None, 10, 7, 21)
        # curr_image = cv2.fastNlMeansDenoising(curr_image, None, 10, 7, 21)
        # ref_polar = cv2.fastNlMeansDenoising(ref_polar, None, 10, 7, 21)
        # curr_polar = cv2.fastNlMeansDenoising(curr_polar, None, 10, 7, 21)

        # curr_polar_init = curr_polar
        init_angle_shift = 0

        # ## Need my own implementation of radon transform
        # # because rotation axis is about the bottom of the image


        # Concatenate zeros to the bottom of the image of the same size as the top
        curr_image_conc = np.concatenate((curr_image, np.zeros_like(curr_image)), axis=0)
        ref_image_conc = np.concatenate((ref_image, np.zeros_like(ref_image)), axis=0)

        # Make the edges of the image black
        curr_image_conc[0:3, :] = 0
        curr_image_conc[-3:, :] = 0
        curr_image_conc[:, 0:3] = 0
        curr_image_conc[:, -3:] = 0
        ref_image_conc[0:3, :] = 0
        ref_image_conc[-3:, :] = 0
        ref_image_conc[:, 0:3] = 0
        ref_image_conc[:, -3:] = 0


        # Binarize images
        # curr_image_conc = cv2.threshold(curr_image_conc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # ref_image_conc = cv2.threshold(ref_image_conc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        theta = np.arange(180)
        sin_curr = radon(curr_image_conc)
        sin_ref = radon(ref_image_conc)


        shift = self.fourier_registration(
            sin_ref, sin_curr, polar=True
        )

        angle_shift = -shift

        #print(f"Detected POLAR subpixel offset (, x): {shift}")
        # angle_shift = -(shift[1]) #-theta[0])
          # 130 degrees is azimuth fov of sonar. This is a linear
        # approximation of the angle shift
        print(f"[Radon] Delta angle (degrees): {angle_shift}")
        
        # Now rotate the image by the angle shift + init_angle_shift (they are in different frames so init is negative)
        M = cv2.getRotationMatrix2D(
            (curr_image.shape[1] / 2, curr_image.shape[0]), -angle_shift - init_angle_shift, 1
        )
        rotated_image = cv2.warpAffine(
            curr_image, M, (curr_image.shape[1], curr_image.shape[0])
        )
        
        if self.blurred_mask is None:

            fan_mask = create_fan_mask(curr_image.shape, 25)
            blurred_mask = cv2.blur(fan_mask, (19, 19))

        rotated_mask = cv2.warpAffine(
            blurred_mask, M, (curr_image.shape[1], curr_image.shape[0])
        )
        # cv2.imshow('rotated image', np.array(rotated_image*rotated_mask, dtype=np.uint8))
        # cv2.imshow('rotated mask', np.array(prev_image*blurred_mask, dtype=np.uint8))
        # cv2.imshow('rotated mask')
        # shift, error, diffphase = phase_cross_correlation(prev_image, rotated_image, disambiguate=True)


        rotated_image_init = rotated_image
        rotated_mask_init = rotated_mask

        shift = FourierRegistration.fourier_registration(
            ref_image,
            rotated_image_init,
            mask=blurred_mask,
            mask_2=rotated_mask_init,
        )
        # print(f"Detected subpixel CUSTOM offset (y, x): {shift_c}")
        print(f"[Radon] Delta Shift (pixels): { np.array([-shift[0], shift[1]])}")

        # Return \theta_ab and shift following x forward y right convention (sonar convention)

        total_angle_shift = angle_shift
        total_shift = np.array([-shift[0], shift[1]], dtype=np.float32)

        print(f"[Radon] Angle (degrees): {total_angle_shift}")
        print(f"[Radon] Shift (pixels): {total_shift}")

        pose = SE2.from_components(
            SO2.Exp(np.deg2rad(total_angle_shift)),
            np.array([total_shift[0], total_shift[1]], dtype=np.float32)
        )

        return pose  # shift in pixels, x right, y down
    

    @staticmethod
    def fourier_registration(image1, image2, mask=None, mask_2 = None, polar=False):
        if mask is None:
            mask = np.ones(image1.shape)
            #zeros on the edges
            mask[0:3, :] = 0
            mask[-3:, :] = 0
            mask[:, 0:3] = 0
            mask[:, -3:] = 0

            mask = cv2.GaussianBlur(mask, (15, 15), 0)
        if mask_2 is None:
            mask_2 = np.ones(image2.shape)
            mask2 = np.ones(image2.shape)
            #zeros on the edges
            mask2[0:3, :] = 0
            mask2[-3:, :] = 0
            mask2[:, 0:3] = 0
            mask2[:, -3:] = 0

            mask2 = cv2.GaussianBlur(mask2, (15, 15), 0)
        image1 = image1 * mask
        image2 = image2 * mask_2
        
        #convert image from 8U to 32F
        ref_ = image1.astype(np.float32) 
        curr_ = image2.astype(np.float32) 

        # (y,x), response = cv2.phaseCorrelate(curr_, ref_)
        # print("OPENCV SHIFT (x, y):", (-x, y))

        # return (x, y)

        # M = cv2.getOptimalDFTSize(ref_.shape[0])
        # N = cv2.getOptimalDFTSize(ref_.shape[1])

        # if(M != ref_.shape[0] or N != ref_.shape[1]):
            
        #         padref = cv2.copyMakeBorder(ref_, 0, M - ref_.shape[0], 0, N - ref_.shape[1], cv2.BORDER_CONSTANT, 0)
        #         padcur = cv2.copyMakeBorder(curr_, 0, M - curr_.shape[0], 0, N - curr_.shape[1], cv2.BORDER_CONSTANT, 0)
               
        # else:   
        #         padref = ref_
        #         padcur = curr_
            
        # f1 = cv2.dft(padref, flags=cv2.DFT_REAL_OUTPUT)
        # f2 = cv2.dft(padcur, flags=cv2.DFT_REAL_OUTPUT)

        # Compute the 2D Fourier Transform of the images
        oned_ft_1 = []
        oned_ft_2 = []
        for i in range(image1.shape[1]):
            ft = np.fft.fft(image1[:, i])
            oned_ft_1.append(np.abs(ft))
            ft = np.fft.fft(image2[:, i])
            oned_ft_2.append(np.abs(ft))

        oned_ft_1 = np.array(oned_ft_1)
        oned_ft_2 = np.array(oned_ft_2)

        # plt.imshow(oned_ft_1.T, cmap='gray')
        # plt.title("1D Fourier Transform along theta of Image 1")
        # plt.show()
        # With this representation, compute 1D cross correlation somehow?
        corr = np.zeros(oned_ft_1.shape[0], dtype=np.complex64)
        for i in range(oned_ft_1.shape[1]):
            f1 = np.fft.fft(oned_ft_1[:, i])
            f2 = np.fft.fft(oned_ft_2[:, i])
            cps = f1 * np.conj(f2) / (np.abs(f1) * np.abs(f2))
            corr += np.fft.ifft(cps)

        ang = np.argmax(corr)
        # print(ang)
        # plt.plot(np.abs(corr))
        # plt.title("Cross-correlation")
        # plt.show()

        if ang > corr.shape[0] // 2:
            ang -= corr.shape[0]
        # print(ang)
        return ang


        # sinofft = np.abs(np.fft.rfft(image1, axis=0))
        # sinofft_rows = sinofft.shape[0]
        # sinofft1 = sinofft[:sinofft_rows // 2, :]

        # sinofft = np.abs(np.fft.rfft(image2, axis=0))
        # sinofft_rows = sinofft.shape[0]
        # sinofft2 = sinofft[:sinofft_rows // 2, :]


        # Fq = np.fft.rfft(sinofft1, axis=0)  # fft along theta axis
        # Fn = np.fft.rfft(sinofft2, axis=0)
        # corrmap_2d = np.fft.irfft(Fq * np.conj(Fn), axis=0)

        # corrmap = np.sum(corrmap_2d, axis=-1)
        # maxval = np.max(corrmap)

        return (y, x)
    

