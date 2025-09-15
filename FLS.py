import numpy as np
import pymlg as mlg
import dataTypes
import models
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import states


class Data(dataTypes.Measurement):
    """
    Data container for a FLS reading.
    """

    def __init__(
        self,
        uv: np.ndarray,
        stamp: float,
        model: "models.MeasurementModel",
        ids: str = None,
    ):
        """
        Parameters
        ----------
        uv : np.ndarray
            feature location reading in pixel coordinates
        stamp : float
            Timestamp of the reading
        model : MeasurementModel, optional
            FLS model for this measurement, by default None
        """
        if type(model) != Model:
            raise TypeError("model must be a FLSModel")
        super().__init__(dof=2, stamp=stamp)
        self.value = np.array(uv).ravel()
        self.model = model
        self.stamp = stamp
        self.ids = ids

    def __repr__(self):
        return f"FLS(stamp={self.stamp}, value={self.value.ravel()})"

    def plus(self, w: np.ndarray) -> "Data":
        """
        Modifies the FLS data. This is used to add noise to the FLS data.

        Parameters
        ----------
        w : np.ndarray
            Value to add to the uv reading

        Returns
        -------
        FLS.Data
            Modified FLS data
        """
        new = self.copy()
        new.value = new.value + w.ravel()
        return new

    def copy(self) -> "Data":
        """
        Returns a copy of the FLS data.

        Returns
        -------
        FLS.Data
            Copy of the DVL data
        """
        return Data(self.value, self.stamp, self.model, self.ids)

    @staticmethod
    def random() -> "Data":
        return Data(np.random.normal(size=2), 0.0, Model(0, 0, 0, 0, 0))


class Model(models.MeasurementModel):
    """
    Forward Looking Sonar measurement model yields pixel measurement given a landmark position
    relative to the sensor frame.
    """

    type = "exteroceptive"

    def __init__(
        self,
        r_min: float,
        r_max: float,
        azimuth_fov: float,
        elevation_fov: float,
        width_image: int,
        sigma: float = 1.0,
        freq: int = None,
        T_bs: np.ndarray = np.identity(4),
    ):
        """_summary_

        Parameters
        ----------
        r_min : float
            Minimum range
        r_max : float
            Maximum range
        azimuth_fov : float
            Azimuth angle
        elevation_fov : float
            Elevation angle
        width_image : int
            Image width
        sigma : float, optional
            FLS noise in pixels, by default 1.0
        freq : int, optional
            Frequency of the FLS, by default None
        T_bs : mlg.SE3, optional
            Transformation matrix, by default np.identity(4)
        """
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.azimuth_fov = float(azimuth_fov)
        self.elevation_fov = float(elevation_fov)
        self.width_image = int(width_image)
        self._covariance = np.eye(2) * np.atleast_2d(sigma) ** 2
        self.freq = freq
        self.T_bs = T_bs

    def __repr__(self):
        return f"FLS {self.__class__.__name__}"

    @property
    def intrinsics(self) -> np.ndarray:
        """
        Returns the intrinsic matrix Q that
        projects 3d point in sonar frame to sonar image plane.
        """
        return (
            1.0055
            / 2
            * np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ]
            )
        )

    def evaluate(self, x: "states.CompositeState") -> np.ndarray:
        """
        Evaluates the measurement model for the landmark state.

        Parameters
        ----------
        x : uvnav.states.CompositeState
            Current state

        Returns
        -------
        np.ndarray
            Expected measurement
        """
        # The pose is always assumed to be the first element
        # TODO: is there a better way to do this? The
        # Measurement class already hold on to the IDs of these two
        # states
        pose: "states.SE3State" = x.value[0]
        landmark: "states.VectorState" = x.value[1]
        r_ps_s = self.resolve_landmark_in_local_frame(pose, landmark)
        y_check = self.project(r_ps_s)
        return y_check

    def resolve_landmark_in_local_frame(
        self,
        pose: "states.SE3State",
        r_pw_a: "states.VectorState",
    ):
        """
        Resolves a landmark with position r_pw_a in the sonar frame.

        Parameters
        ----------
        pose : uvnav.states.SE3State
            Current pose
        r_pw_a : uvnav.states.VectorState
            Landmark position

        Returns
        -------
        np.ndarray
            Landmark position in the sonar frame
        """
        r_pw_a = r_pw_a.value.reshape((-1, 1))
        C_bs = self.T_bs[0:3, 0:3]
        C_ab = pose.attitude
        r_zw_a = pose.position.reshape((-1, 1))
        r_sz_b = (self.T_bs[0:3, 3]).reshape((-1, 1))

        r_ps_s = C_bs.T @ (C_ab.T @ (r_pw_a - r_zw_a) - r_sz_b)
        return r_ps_s

    def project(self, r_ps_s: np.ndarray) -> np.ndarray:
        """
        Returns pixel measurement of landmark.

        Parameters
        ----------
        r_ps_s : np.ndarray
            Landmark position in the sonar frame

        Returns
        -------
        np.ndarray
            Pixel measurement of the landmark
        """
        x_, y_, z_ = r_ps_s

        r = np.linalg.norm(r_ps_s)
        elev = np.arcsin(z_ / r)
        cos_elev = np.cos(elev)

        gamma = self.width_image / (
            2.0 * self.r_max * np.sin(np.deg2rad(self.azimuth_fov) / 2.0)
        )

        u = gamma * y_ / cos_elev + self.width_image / 2.0
        v = gamma * (self.r_max - x_ / cos_elev)
        return np.array([u, v])
 
    def jacobian_projection(self, r_ps_s: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian of the projection function, given a landmark in the sonar frame.

        Parameters
        ----------
        r_ps_s : np.ndarray
            Landmark position in the sonar frame

        Returns
        -------
        np.ndarray
            Jacobian of the projection
        """
        x_, y_, z_ = r_ps_s

        r_ = np.linalg.norm(r_ps_s)
        r_small = np.linalg.norm(r_ps_s[0:2])

        gamma = self.width_image / (
            2.0 * self.r_max * np.sin(np.deg2rad(self.azimuth_fov) / 2.0)
        )

        jacob_proj = np.array(
            [
                [
                    -gamma * x_ * y_ * pow(z_, 2) / (pow(r_small, 3) * r_),
                    gamma
                    * (
                        pow(x_, 4)
                        + pow(y_, 4)
                        + 2 * pow(x_, 2) * pow(y_, 2)
                        + pow(x_, 2) * pow(z_, 2)
                    )
                    / (pow(r_small, 3) * r_),
                    gamma * (y_ * z_ / (r_small * r_)),
                ],
                [
                    -gamma
                    * (
                        pow(x_, 4)
                        + pow(y_, 4)
                        + 2 * pow(x_, 2) * pow(y_, 2)
                        + pow(y_, 2) * pow(z_, 2)
                    )
                    / (pow(r_small, 3) * r_),
                    gamma * x_ * y_ * pow(z_, 2) / (pow(r_small, 3) * r_),
                    -gamma * (x_ * z_ / (r_small * r_)),
                ],
            ]
        )
        return jacob_proj

    def is_measurement_valid(self, uv: np.ndarray) -> bool:
        """
        Checks if measurement is valid.

        Parameters
        ----------
        uv : np.ndarray
            Pixel measurement

        Returns
        -------
        bool
            True if the measurement is valid
        """
        # Convert pixel measurement to bearing and range (in that order)
        az_range = self.convert_to_bearing_range(uv)
        return (
            (abs(az_range[0]) < np.deg2rad(self.azimuth_fov / 2.0))
            and (az_range[1] > self.r_min)
            and (az_range[1] < self.r_max)
        )

    def is_landmark_in_fov(
        self, pose: "states.SE3State", r_pw_a: "states.VectorState"
    ) -> bool:
        """
        Checks if a given landmark is in fov of the sonar.

        Parameters
        ----------
        pose : uvnav.states.SE3State
            Current pose
        r_pw_a : uvnav.states.VectorState
            Landmark position

        Returns
        -------
        bool
            True if the landmark is in the sonar fov
        """
        r_ps_s: np.ndarray = self.resolve_landmark_in_local_frame(pose, r_pw_a)
        r_polar = self.to_polar(r_ps_s)  # az, el, r

        return (
            (abs(r_polar[0]) < np.deg2rad(self.azimuth_fov / 2.0))  # az
            and (r_polar[2] > self.r_min)  # r
            and (r_polar[2] < self.r_max)  # r
            and (abs(r_polar[1]) < np.deg2rad(self.elevation_fov / 2.0))  # el
        )

    def convert_to_bearing_range(self, uv: np.ndarray) -> np.ndarray:
        """
        Converts pixel measurement to bearing and range.

        Parameters
        ----------
        uv : np.ndarray
            Pixel measurement

        Returns
        -------
        np.ndarray
            Bearing and range
        """
        u, v = uv
        gamma = self.width_image / (
            2.0 * self.r_max * np.sin(np.deg2rad(self.azimuth_fov) / 2.0)
        )
        a = (u - self.width_image / 2.0) / gamma
        b = self.r_max - v / gamma
        r = np.linalg.norm(np.array([a, b]))
        az = np.arctan2(a, b)
        return np.array([az, r])

    def jacobian(self, x: "states.CompositeState") -> np.ndarray:
        """
        Evaluates the measurement model Jacobian.

        Parameters
        ----------
        x : uvnav.states.CompositeState
            Current state

        Returns
        -------
        np.ndarray
            Measurement model Jacobian
        """
        pose: "states.SE3State" = x.value[0]
        landmark: "states.VectorState" = x.value[1]

        r_zw_a = pose.position.reshape((-1, 1))
        C_ab = pose.attitude
        C_bc = self.T_bs[0:3, 0:3]
        r_pw_a = landmark.value.reshape((-1, 1))
        r_cz_b = self.T_bs[0:3, 3].reshape((-1, 1))

        y = C_ab.T @ (r_pw_a - r_zw_a)
        p_is_s = C_bc.T @ (y - r_cz_b)

        jacob_proj = self.jacobian_projection(p_is_s)

        # Compute Jacobian of measurement model with respect to the state
        if pose.direction == "right":
            pose_jacobian = pose.jacobian_from_blocks(
                attitude=-jacob_proj @ C_bc.T @ mlg.SO3.odot(y),
                position=-jacob_proj @ C_bc.T,
            )
        elif pose.direction == "left":
            pose_jacobian = pose.jacobian_from_blocks(
                attitude=-jacob_proj @ C_bc.T @ C_ab.T @ mlg.SO3.odot(r_pw_a),
                position=-jacob_proj @ C_bc.T @ C_ab.T,
            )

        # Compute jacobian of measurement model with respect to the landmark
        landmark_jacobian = jacob_proj @ pose.attitude.T

        # Build full Jacobian
        state_ids = [state.id for state in x.value]
        jac_dict = {}
        jac_dict[state_ids[0]] = pose_jacobian
        jac_dict[state_ids[1]] = landmark_jacobian
        return x.jacobian_from_blocks(jac_dict)

    def covariance(self, x: "states.CompositeState") -> np.ndarray:
        """
        Returns the covariance :math:`\mathbf{R}` associated with additive Gaussian noise.

        Parameters
        ----------
        x : uvnav.states.CompositeState
            Current state

        Returns
        -------
        np.ndarray
            Measurement noise covariance
        """
        return self._covariance

    def to_polar(self, r_ps_s: np.ndarray) -> np.ndarray:
        """
        Converts 3D cartesian coordinates to polar coordinates.

        Parameters
        ----------
        r_ps_s : np.ndarray
            3D cartesian coordinates

        Returns
        -------
        np.ndarray
            Polar coordinates
        """
        az = np.arctan2(r_ps_s[1], r_ps_s[0])[0]
        el = np.arctan2(r_ps_s[2], np.linalg.norm(r_ps_s[0:2]))[0]
        r = np.linalg.norm(r_ps_s)
        return np.array([az, el, r])

    def to_cartesian(self, y: np.ndarray) -> np.ndarray:
        """
        Converts 3D polar coordinates to cartesian coordinates.

        Parameters
        ----------
        y : np.ndarray
            3D polar coordinates

        Returns
        -------
        np.ndarray
            3D cartesian coordinates
        """
        az, el, r = y
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return np.array([x, y, z])
