
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import models


class Measurement(ABC):
    """
    An abstract data container that holds a measurement value.
    The value is specified in inherited classes.
    """

    __slots__ = ["dof", "stamp", "value", "model"]
    dof: int
    stamp: float
    value: np.ndarray
    model: "models.MeasurementModel"

    def __init__(self, dof: int, stamp: float = None):
        """
        Parameters
        ----------
        dof : int
            Degrees of freedom of the measurement
        stamp : float, optional
            Timestamp of the measurement, by default None
        """
        self.dof = dof
        self.stamp = stamp

    @abstractmethod
    def plus(self, w: np.ndarray) -> "Measurement":
        """
        Generic addition operation to modify the internal value.

        Parameters
        ----------
        w : np.ndarray
            Noise vector

        Returns
        -------
        Measurement
            Modified measurement
        """
        pass

    @abstractmethod
    def copy(self) -> "Measurement":
        """
        Creates a deep copy of the object.

        Returns
        -------
        Measurement
            Copy of the measurement
        """
        pass


class RelativeMotionIncrement(Measurement):
    """
    Relative motion increment between two states.
    RMI becomes a new input to the system.
    """

    __slots__ = ["dof", "stamps", "value", "covariance", "id"]
    dof: int
    stamps: list
    value: np.ndarray
    covariance: np.ndarray
    id: str

    def __init__(self, dof: int):
        #:int: Degrees of freedom of the RMI
        self.dof = dof

        #:List[float, float]: the two timestamps i, j associated with the RMI
        self.stamps = [None, None]

        #:np.ndarray: the value of the RMI
        self.value = None

        #:np.ndarray: the covariance matrix of the RMI
        self.covariance = None

        #:Any: an ID associated with the RMI
        self.state_id = None

    @abstractmethod
    def increment(self, u: "Measurement", dt: float):
        """
        Increment the RMI.

        Parameters
        ----------
        u : dataTypes.Measurement
            Measurement to use for the increment
        dt : float
            Time increment
        """
        pass

    @abstractmethod
    def new(self) -> "RelativeMotionIncrement":
        """
        Create a new RMI.

        Returns
        -------
        RelativeMotionIncrement
            New RMI
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the RMI.
        """
        pass

    def symmetrize(self):
        """
        Symmetrize the covariance matrix of the RMI.
        """
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
