import numpy as np
import cv2
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})


def plot_polar(data: np.ndarray, fov_degrees: float, min_range: float, max_range: float, detected_tags, title: str='', real_data: bool = False) -> None:

    # Get theta and range values for the image
    min_theta, max_theta = np.array([-fov_degrees/2, fov_degrees/2])
    ranges = np.linspace(max_range, min_range, data.shape[0])
    if not real_data:
        thetas = np.deg2rad(np.linspace(max_theta, min_theta, data.shape[1]))
    else:
        thetas = np.array([-65  , -64.05, -63.13, -62.24, -61.38, -60.54, -59.72, -58.93,
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
        thetas = np.flip(np.deg2rad(thetas))

    # Create a meshgrid so we can plot the 2D function
    thetasMesh, rangesMesh = np.meshgrid(thetas, ranges)
    
    # Create polar plot and set image values.
    figp, axp = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
    axp.pcolormesh(thetasMesh, rangesMesh, data, cmap = 'magma')

    # Adjust origin to be in the center of the image
    axp.set_theta_zero_location('N')

    # Set bounds for theta and range
    axp.set_thetamin(min_theta)
    axp.set_thetamax(max_theta)
    axp.set_rmax(max_range)
    axp.set_theta_direction(-1)

    tick_angles = np.linspace(-65, 65, num=11)  # [-65, -43.3, -21.7, 0, 21.7, 43.3, 65]
    axp.set_thetagrids(tick_angles, labels=[f"{int(a)}°" for a in tick_angles])
    # ...existing code...

    # Set title and turn off the grid
    # axp.set_title(title)
    axp.grid(True, color='gray', alpha=0.3, linewidth=0.7)  # Make grid lines faint

    plt.tight_layout()
    plt.savefig("/home/arturo/Documents/phd/icra_paper/ieeeconf/figures/polar_plot.png", dpi=300, bbox_inches='tight')
    plt.show()



def create_fan_mask(image_shape, start_angle):
    """
    Create a fan-shaped mask.

    Args:
        image_shape (tuple): Shape of the image (height, width).
        center (tuple): Center of the fan (x, y).
        radius (int): Radius of the fan.
        start_angle (float): Start angle of the fan in degrees.
        end_angle (float): End angle of the fan in degrees.

    Returns:
        numpy.ndarray: Fan-shaped mask.
    """

    (h, w) = image_shape
    center = (w//2, h)

    img = np.zeros(image_shape, dtype=np.float32)
    cv2.circle(img, center, h, 1, -1)
    # cv2.imshow('Circle', img)
    # Calculate the points for the fan shape

    start_angle_rad = np.deg2rad(start_angle)

    points = []
    points.append(center)
    points.append((w, h))
    points.append((w, h * (1- np.sin(start_angle_rad))))

    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    # Draw the fan shape on the mask
    cv2.fillPoly(img, [points], 0)

    points = []
    points.append(center)
    points.append((0, h))
    points.append((0, h * (1- np.sin(start_angle_rad))))

    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    # Draw the fan shape on the mask
    cv2.fillPoly(img, [points], 0)

    return img

def resize_image(image, scale_factor):
    """
    Resize the image by a given scale factor.

    Args:
        image (numpy.ndarray): The input image.
        scale_factor (float): The factor by which to scale the image.

    Returns:
        numpy.ndarray: The resized image.
    """
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    resized_image = cv2.resize(image, new_size)
    return resized_image


def getPixelValue(image, uv):
    "Apply bilinear interpolation to get pixel value at non-integer coordinates."
    x, y = uv
    xx = x- int(x)
    yy = y- int(y)



    I11 = image[int(y), int(x)]
    I12 = image[int(y), int(x) + 1]
    I21 = image[int(y) + 1, int(x)]
    I22 = image[int(y) + 1, int(x) + 1]

    I = (1-xx) * (1-yy) * I11 + xx * (1-yy) * I12 + (1-xx) * yy * I21 + xx * yy * I22
    return I

def getPatternPixelValue(image, uv):
    x, y = uv

    x2 = x + 0
    y2 = y - 2

    x3 = x - 1
    y3 = y - 1

    x4 = x + 1
    y4 = y - 1

    x5 = x - 2
    y5 = y

    x6 = x + 2
    y6 = y

    x7 = x - 1
    y7 = y + 1

    x8 = x + 1
    y8 = y + 1

    x9 = x
    y9 = y + 2
    
    v1 = getPixelValue(image, (x, y))
    v2 = getPixelValue(image, (x2, y2))
    v3 = getPixelValue(image, (x3, y3))
    v4 = getPixelValue(image, (x4, y4))
    v5 = getPixelValue(image, (x5, y5))
    v6 = getPixelValue(image, (x6, y6))
    v7 = getPixelValue(image, (x7, y7)) 
    v8 = getPixelValue(image, (x8, y8))
    v9 = getPixelValue(image, (x9, y9))

    return 0.5 * v1 + (0.5 / 8.0) * (v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9)


def getGradient(image, uv):
    x, y = uv

    x2 = x
    y2 = y - 1

    x3 = x - 1
    y3 = y - 1

    x4 = x + 1
    y4 = y - 1

    x5 = x - 1
    y5 = y

    x6 = x + 1
    y6 = y

    x7 = x - 1
    y7 = y + 1

    x8 = x + 1
    y8 = y + 1

    x9 = x
    y9 = y + 1
    
    v1 = getPixelValue(image, (x, y))
    v2 = getPixelValue(image, (x2, y2))
    v3 = getPixelValue(image, (x3, y3))
    v4 = getPixelValue(image, (x4, y4))
    v5 = getPixelValue(image, (x5, y5))
    v6 = getPixelValue(image, (x6, y6))
    v7 = getPixelValue(image, (x7, y7)) 
    v8 = getPixelValue(image, (x8, y8))
    v9 = getPixelValue(image, (x9, y9))

    neighborhood = np.array([[v3, v2, v4],
                             [v5, v1, v6],
                             [v7, v9, v8]])

    kern_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kern_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


    grad_x = np.sum(kern_x * neighborhood)
    grad_y = np.sum(kern_y * neighborhood)

    return np.array([grad_x, grad_y]).reshape((1, -1))  # Reshape to (1, 2) for consistency

def to_cartesian(polar_img, min_range=0.2, max_range=3.0, azimuth_fov = 130):
    

    if min_range > 0:
        # Pad zeros to the top of the image
        range_res =  polar_img.shape[0]/ (max_range - min_range) # pixels per meter
        num_padding = int(np.round(min_range * range_res)) # Add the number of pixels corresponding to empty space of min range
   
        polar_img = cv2.copyMakeBorder(polar_img, num_padding, 0, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
        # polar_img = np.pad(polar_img, ((num_padding, 0), (0, 0)), mode='constant', constant_values=0)

   
    newmapx, newmapy = mappingPolar(polar_img, azimuth_fov=azimuth_fov)
    out = cv2.remap(polar_img.T, newmapx, newmapy, cv2.INTER_CUBIC,
            cv2.BORDER_CONSTANT)

    return out

def mappingPolar(rect, azimuth_fov=130):

    ## if min range is above 0, then we pad zeros to the top of the image
    
    #thetas = get_thetas(msg)
    #thetas.reverse()
    # ranges = list(msg.ranges)
    #ranges.reverse()
    nRanges = rect.shape[0]
    #theta_min = thetas[0]
    #theta_max = thetas[-1]

    theta_min = np.deg2rad(-azimuth_fov / 2.0)
    theta_max = np.deg2rad(azimuth_fov / 2.0)



    minusWidth = np.floor(nRanges * np.sin(theta_min))
    plusWidth = np.ceil(nRanges * np.sin(theta_max))
    width = int(plusWidth - minusWidth)

    originx = int(abs(minusWidth))


 

    mapx = np.zeros((nRanges, width), dtype=np.float32)
    mapy = np.zeros((nRanges, width), dtype=np.float32)

    db = (theta_max- theta_min) / rect.shape[1]  # msg.image.beam_count

    for x in range(width):
        for y in range(nRanges):
            #This creates a map that maps the destination image to the source image
            # so it takes pixel positions and gets range and bearing, which 
            # is what we have in the rect image.
            dx = x - originx
            dy = nRanges - y

            rangey = np.sqrt(dx * dx + dy * dy)
            azimuth = np.arctan2(dx, dy)

            xp = rangey

            #todo This linear algorithm is not robust if the azimuths
            # are non-linear.   Should implement a real interpolation...
            yp = (azimuth - theta_min) / db  #beam number

            mapx[y,x] = xp
            mapy[y,x] = yp


    return mapx, mapy#cv2.convertMaps(newma, None, _scMap1, _scMap2, cv2.16SC2)

from scipy.spatial.transform import Rotation as R
from pymlg.numpy import SE3
from states import SE3State

def load_gt(gt_path : Path):
    """
    Load ground truth data from a file.

    Args:
        gt_path (str or Path): Path to the ground truth file.

    Returns:
        dict: Dictionary containing ground truth data.
    """
    gt_data = []
    i = 0
    with open(gt_path, 'r') as f:
        for line in f:
            if i == 0:
                i += 1
                continue
            row = line.split(",")
            
            trans_xyz = np.array([float(row[1]), float(row[2]), float(row[3])]).reshape(-1, 1)
            quat_xyzw = np.array([float(row[5]), float(row[6]), float(row[7]), float(row[4])])
            time = float(row[0])

            # Convert quaternion to rotation matrix
            Rot = R.from_quat(quat_xyzw).as_matrix()
            
            pose = SE3.from_components(Rot,  trans_xyz)

            pose_state = SE3State(pose, time, i)

            gt_data.append(pose_state)
            
            i += 1
    return gt_data

def convert_se3_estimate_to_planar_estimates(pose: np.ndarray):
    
    vector = SE3.Log(pose).ravel()
        #print(f"ICP Registration vector: {vector}")
    angle_shift = np.rad2deg(vector[2])  # angle in degrees
    
    # The translation is in meters, convert to pixels
    translation_shift = np.array([pose[0,3], pose[1,3]], dtype=np.float32)  # translation in meters
    
    return angle_shift, translation_shift

import xml.etree.ElementTree as ET

def readAgisoftTrajectory(
    fpath: str, cam_param: dict, mlg_direction: str = "right"
) -> list[SE3State]:
    """
    Reads an Agisoft trajectory file in .xml and returns a list of SE3State objects.

    The XML file should contain camera transforms in the following format:
    <camera id="0" sensor_id="2" component_id="0", label="filename">
        <transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</transform>
        <orientation>1</orientation>
    </camera>

    "sensor_id" 2 is the left camera, and "sensor_id" 3 is the right camera.

    Parameters
    ----------
    fpath : str
        Path to the Agisoft trajectory XML file.
    mlg_direction : str
        Direction of the MLG (e.g., "right", "left")

    Returns
    -------
    list[SE3State]
        A list of SE3State objects representing the camera poses.
    """
    tree = ET.parse(fpath)
    root = tree.getroot()

    # Find the <scale> element
    scale_element = root.find(".//scale")
    if scale_element is not None and scale_element.text:
        scale = float(scale_element.text)

    # Iterate through all <camera> elements
    pose_list: list[SE3State] = []
    for camera in root.findall(".//camera"):
        camera_id = camera.get("id")
        camera_id = int(camera_id) if camera_id.isdigit() else camera_id
        sensor_id = int(camera.get("sensor_id"))
        if sensor_id == 0:
            T_bcam = np.array(cam_param["cam0"]["T_body_sensor"])
        elif sensor_id == 1:
            T_bcam = np.array(cam_param["cam1"]["T_body_sensor"])
        else:
            raise ValueError(f"Invalid sensor_id {sensor_id}. Expected 2 or 3.")
        label = camera.get("label")
        transform = camera.find("transform")

        if transform is not None and transform.text:
            # Parse the transform matrix
            transform_matrix = np.array(
                list(map(float, transform.text.split()))
            ).reshape(4, 4)
            transform_matrix[:3, -1] *= scale
            pose_list.append(
                SE3State(
                    value=T_bcam @ transform_matrix @ SE3.inverse(T_bcam),
                    stamp=datetime2unix(
                        label.split("SYSTEM_")[1].split("_")[0].replace("-", "")
                    ),
                    id=camera_id,
                    direction=mlg_direction,
                )
            )
    return pose_list

def correct_timestamps(gt_states, cam_file_csv):
    """
    Corrects the timestamps of the ground truth states based on a camera file.

    Parameters
    ----------
    gt_states : list[SE3State]
        List of ground truth states to be corrected.
    cam_file_csv : str
        Path to the camera file in CSV format.

    Returns
    -------
    list[SE3State]
        List of corrected ground truth states.
    """
    import pandas as pd

    cam_data = pd.read_csv(cam_file_csv)
    # cam_data['#timestamp'] = pd.to_datetime(cam_data['#timestamp'], unit='s')
    filenames = cam_data['image_file_x'].values
    
    file_timestamps = [filename.split('SYSTEM_')[1].split("_")[0] for filename in filenames]

    file_timestamps = [timestamp.replace("-", "") for timestamp in file_timestamps]

    unix_timestamps = np.array([datetime2unix(timestamp.split(".")[0])+int(timestamp.split(".")[1])/1e6 for timestamp in file_timestamps])
    # obtain unix timestamps from the camera file name
    


    for state in gt_states:
        timestamp = state.stamp
        # Find the closest timestamp in the unix_timestamps array and its index,
        # then asign to the corresponding value in cam_data['#timestamp']

        closest_index = (unix_timestamps - timestamp).argmin()

        closest_time = cam_data['#timestamp'].iloc[closest_index]

        state.stamp = closest_time

    return gt_states




def datetime2unix(datetime: str, time_zone: str = "UTC") -> float:
    """
    Converts a datetime string in the format to unix timestamp.
    The image file name is expected to be in the format:

        {filename}_{timestamp}_{frame_number}.jpg

    where timestamp is in the dateteim format:

        YYYY-MM-DDTHHMMSS.MMMSSS

    The timestamp is converted to epoch time.

    Parameters
    ----------
    datetime : str
        Datetime string in the format "YYYY-MM-DDTHHMMSS.MMMSSS".

    Returns
    -------
    float
        Unix timestamp corresponding to the input datetime.
    """
    import ciso8601
    import pytz

    # convert datetime to epoch time
    return (
        ciso8601.parse_datetime(datetime)
        .replace(tzinfo=pytz.timezone(time_zone))
        .timestamp()
    )



def is_mostly_black(image, max_threshold=10, mean_threshold=1, min_non_black_percent=1):
    """Check if image is mostly black using multiple criteria"""
    # Check 1: Maximum pixel value
    if image.max() < max_threshold:
        return True
    
    # Check 2: Mean pixel value
    # if np.mean(image) < mean_threshold:
    #     return True
    
    # Check 3: Percentage of non-black pixels
    non_black_pixels = np.sum(image > 20)
    percentage_non_black = (non_black_pixels / image.size) * 100
    if percentage_non_black < min_non_black_percent:
        return True
    
    return False


import numpy as np
from scipy.interpolate import interp1d
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    import states

def state_interp(
    query_stamps: float | list[float],
    state_list: list["states.State"] | list["states.StateWithCovariance"],
    method: str = "linear",
    covariance: bool = False,
    verbose: bool = False,
    degree: int = 10,
) -> Union[
    "states.State",
    list["states.State"],
    "states.StateWithCovariance",
    list["states.StateWithCovariance"],
    tuple,
]:
    """
    Performs "linear" (geodesic) interpolation between ``State`` objects. Multiple
    interpolations can be performed at once in a vectorized fashion. If the
    query point is out of bounds, the end points are returned.

    ..code-block:: python

        x_data = [SE3State.random(stamp=i) for i in range(10)]
        x_query = [0.2, 0.5, 10]
        x_interp = state_interp(x_query, x_data)

    Parameters
    ----------
    query_stamps : float or object with ``.stamp`` attribute (or Lists thereof)
        Query stamps. Can either be a float, or an object containing a ``stamp``
        attribute. If a list is provided, it will be treated as multiple query
        points and the return value will be a list of ``states.State`` objects.
    state_list : list[states.State] or list[states.StateWithCovariance]
        List of ``states.State`` or ``states.StateWithCovariance`` objects to interpolate

    Returns
    -------
    ``states.State`` or list[``states.State``]
        The interpolated state(s).

    Raises
    ------
    TypeError
        If query point is not a float or object with a ``stamp`` attribute.
    """

    # Handle input
    if isinstance(query_stamps, list):
        single_query = False
    elif isinstance(query_stamps, np.ndarray):
        single_query = False
    elif isinstance(query_stamps, float):
        query_stamps = [query_stamps]
        single_query = True
    else:
        pass

    query_stamps = query_stamps.copy()
    for i, stamp in enumerate(query_stamps):
        if not isinstance(stamp, (float, int)):
            if hasattr(stamp, "stamp"):
                stamp = stamp.stamp
                query_stamps[i] = stamp
            else:
                raise TypeError(
                    "Stamps must be of type float or have a stamp attribute"
                )

    # Get the indices of the states just before and just after.
    query_stamps = np.array(query_stamps)
    state_list = np.array(state_list)
    stamp_list = [state.stamp for state in state_list]
    stamp_list.sort()
    stamp_list = np.array(stamp_list)

    if method == "linear":
        idx_middle = np.interp(
            query_stamps, stamp_list, np.array(range(len(stamp_list)))
        )
        idx_lower = np.floor(idx_middle).astype(int)
        idx_upper = idx_lower + 1

        before_start = query_stamps < stamp_list[0]
        after_end = idx_upper >= len(state_list)
        inside = np.logical_not(np.logical_or(before_start, after_end))

        # Return endpoint if out of bounds
        idx_upper[idx_upper == len(state_list)] = len(state_list) - 1

        # ############ Do the interpolation #################
        stamp_lower = stamp_list[idx_lower]
        stamp_upper = stamp_list[idx_upper]

        # "Fraction" of the way between the two states
        alpha = np.zeros(len(query_stamps))
        alpha[inside] = np.array(
            (query_stamps[inside] - stamp_lower[inside])
            / (stamp_upper[inside] - stamp_lower[inside])
        ).ravel()

        # The two neighboring states around the query point
        state_lower: list["states.State"] | list["states.StateWithCovariance"] = (
            np.array(state_list[idx_lower]).ravel()
        )
        state_upper: list["states.State"] | list["states.StateWithCovariance"] = (
            np.array(state_list[idx_upper]).ravel()
        )

        # Interpolate between the two states

        if not covariance:
            dx = np.array(
                [
                    s.minus(state_lower[i]).get_value().ravel()
                    for i, s in enumerate(state_upper)
                ]
            )

            out = []
            for i, state in enumerate(state_lower):
                if np.isnan(alpha[i]) or np.isinf(alpha[i]) or alpha[i] < 0.0:
                    raise RuntimeError("wtf")

                state_interp = state.plus(dx[i] * alpha[i])

                state_interp.stamp = query_stamps[i]
                out.append(state_interp)
        else:
            dx = np.array(
                [
                    s.state.minus(state_lower[i].state).get_value().ravel()
                    for i, s in enumerate(state_upper)
                ]
            )

            # P_k = [state.covariance for state in state_upper]
            out = []
            for i, state in enumerate(state_lower):
                if np.isnan(alpha[i]) or np.isinf(alpha[i]) or alpha[i] < 0.0:
                    raise RuntimeError("wtf")

                state_: "states.MatrixLieGroupState" = state.state
                state_interp = state_.plus(dx[i] * alpha[i])

                state_interp.stamp = query_stamps[i]

                # Covariance
                J_1 = state_.group.right_jacobian_inv(dx[i])  # NEED to make this better
                J_2 = state_.group.right_jacobian(dx[i] * alpha[i])

                A = alpha[i] * J_2 @ J_1
                P_t_1 = (
                    (np.eye(state_.dof) - A)
                    @ state.covariance
                    @ (np.eye(state_.dof) - A).T
                )
                P_t_2 = A @ state_upper[i].covariance @ A.T

                P_t = P_t_1 + P_t_2

                out.append([state_interp, P_t])


    elif method == "nearest":
        indexes = np.array(range(len(stamp_list)))
        nearest_state = interp1d(
            stamp_list,
            indexes,
            "nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        state_idx = nearest_state(query_stamps).astype(int)
        out = state_list[state_idx].tolist()

    if single_query:
        out = out[0]

    if verbose:
        return out, (dx, alpha, stamp_lower, stamp_upper)
    else:
        return out
