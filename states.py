

import numpy as np
import pymlg as mlg
from abc import ABC, abstractmethod
from typing import Union


class State(ABC):
    """
    An abstract state :math:`\\mathcal{X}` is an object containing the following attributes:

    - a value of some sort;
    - a certain number of degrees of freedom (dof);
    - ``plus`` and ``minus`` methods that generalize addition and subtracting to this object.

    Optionally, it is often useful to assign a timestamp (``stamp``) and an identifier
    (``id``) to differentiate state instances from others.

    When implementing a new state type, you should inherit from this class as
    shown in the tutorial.

    .. note::
        The ``plus`` and ``minus`` must correspond to each other, in the sense
        that the following must hold:

        .. math::

            \\delta \\mathbf{x} = (\\mathcal{X} \\oplus \\delta \\mathbf{x}) \\ominus \\mathcal{X}

        for any state :math:`\\mathcal{X}` and any perturbation :math:`\\delta \\mathbf{x}`.
    """

    __slots__ = ["value", "dof", "stamp", "id"]

    def __init__(self, value, dof: int, stamp: float = None, id: str = None):
        """
        Parameters
        ----------
        value : Any
            Value of the state. This can be anything, but must be compatible with
            the ``plus`` and ``minus`` methods
        dof : int
            Number of degrees of freedom of the state
        stamp : float, optional
            Timestamp, by default None
        id : str, optional
            State id for description, by default None
        """
        self.value = value
        self.dof = dof
        self.stamp = stamp
        self.id = id

    def __repr__(self):
        value_str = str(self.value).split("\n")
        value_str = "\n".join(["    " + s for s in value_str])
        s = [
            f"{self.__class__.__name__}(stamp={self.stamp}, dof={self.dof}, id={self.id})",
            f"{value_str}",
        ]
        return "\n".join(s)

    @abstractmethod
    def get_value(self) -> np.ndarray:
        """
        Returns the 1D numeric value of the state.
        """
        pass

    @abstractmethod
    def plus(self, x: "State") -> "State":
        """
        A generic ``addition`` operation given a like-state ``x``.
        """
        pass

    @abstractmethod
    def minus(self, x: "State") -> "State":
        """
        A generic ``subtraction`` operation given a like-state ``x``.
        """
        pass

    @abstractmethod
    def dot(self, x: "State") -> "State":
        """
        A generic dot product operation given a like-state ``x``.
        """
        pass

    @abstractmethod
    def copy(self) -> "State":
        """
        Returns a copy of this State instance.
        """
        pass

    @staticmethod
    @abstractmethod
    def random() -> "State":
        """
        Returns an instance of this State with random values.
        """
        pass

    @staticmethod
    @abstractmethod
    def inverse() -> "State":
        """
        Returns an inverse of this State instance.
        """
        pass

    def plus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        """
        Jacobian of the ``plus`` operator. That is, using Lie derivative notation,

        .. math::

            \\mathbf{J} = \\frac{D (\\mathbf{X} \\oplus \\delta \\mathbf{x})}{D \\delta \\mathbf{x}}


        For Lie groups, this is known as the *group Jacobian*.
        """
        return self.plus_jacobian_fd(dx)

    def plus_jacobian_fd(self, dx: np.ndarray, step_size=1e-8) -> np.ndarray:
        """
        Calculates the plus jacobian with finite difference.
        """
        dx_bar = dx
        jac_fd = np.zeros((self.dof, self.dof))
        Y_bar = self.plus(dx_bar)
        for i in range(self.dof):
            dx = np.zeros((self.dof,))
            dx[i] = step_size
            Y: State = self.plus(dx_bar.ravel() + dx)
            jac_fd[:, i] = Y.minus(Y_bar).get_value() / step_size

        return jac_fd

    def minus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        """
        Jacobian of the ``minus`` operator with respect to self.

        .. math::

            \\mathbf{J} = \\frac{D (\\mathbf{X} \\ominus \\delta \\mathbf{X})}{D \\mathbf{X}}

        That is, if ``dx = y.minus(x)`` then this is the Jacobian of ``dx`` with respect to ``y``.
        For Lie groups, this is the inverse of the *group Jacobian* evaluated at
        ``dx = x1.minus(x2)``.
        """
        return self.minus_jacobian_fd(dx)

    def minus_jacobian_fd(self, x: "State", step_size=1e-8) -> np.ndarray:
        """
        Calculates the minus jacobian with finite difference.
        """
        x_bar = x
        jac_fd = np.zeros((self.dof, self.dof))

        y_bar = self.minus(x_bar)
        for i in range(self.dof):
            dx = np.zeros((self.dof,))
            dx[i] = step_size
            y: State = self.plus(dx).minus(x_bar)
            jac_fd[:, i] = (y.get_value() - y_bar.get_value()) / step_size

        return jac_fd


class StateWithCovariance:
    """
    A data container containing a ``State`` object and a covariance array.
    This class can be used as-is without inheritance.
    """

    __slots__ = ["state", "covariance"]

    def __init__(self, state: State, covariance: np.ndarray):
        """
        Parameters
        ----------
        state : State
            A state object, usually representing the mean of a distribution
        covariance : np.ndarray
            A square, symmetric covariance matrix associated with the state

        Raises
        ------
        ValueError
            If the covariance matrix is not square.
        ValueError
            If the covariance matrix does not correspond with the state degrees
            of freedom.
        """
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be an n x n array.")

        if covariance.shape[0] != state.dof:
            raise ValueError("Covariance matrix does not correspond with state DOF.")

        #:navlie.types.State: state object
        self.state = state

        #:numpy.ndarray: covariance associated with state
        self.covariance = covariance

    def __repr__(self):
        return f"StateWithCovariance(stamp={self.stamp})"

    @property
    def stamp(self):
        return self.state.stamp

    @stamp.setter
    def stamp(self, stamp):
        self.state.stamp = stamp

    def symmetrize(self):
        """
        Enforces symmetry of the covariance matrix.
        """
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def copy(self) -> "StateWithCovariance":
        return StateWithCovariance(self.state.copy(), self.covariance.copy())


class VectorState(State):
    """
    A standard vector-based state, with value represented by a 1D numpy array.
    """

    def __init__(self, value: np.ndarray, stamp: float = None, id: str = None):
        """
        Parameters
        ----------
        value : np.ndarray
            Value of the state in 1D numpy array
        stamp : float, optional
            Timestamp, by default None
        id : str, optional
            State id for description, by default None
        """
        value = np.array(value, dtype=np.float64).ravel()
        super(VectorState, self).__init__(
            value=value,
            dof=value.size,
            stamp=stamp,
            id=id,
        )
        self.value: np.ndarray = self.value  # just for type hinting

    def get_value(self) -> np.ndarray:
        return self.value.ravel()

    def plus(self, x: Union[np.ndarray, "VectorState"]) -> "VectorState":
        new = self.copy()
        if isinstance(x, np.ndarray):
            if x.size == self.dof:
                new.value = new.value.ravel() + x.ravel()
            else:
                raise ValueError("Mismatched state size added to VectorState.")
        elif isinstance(x, VectorState):
            if x.dof == self.dof:
                new.value = new.value.ravel() + x.value.ravel()
            else:
                raise ValueError("Mismatched state size added to VectorState.")
        else:
            raise ValueError("Invalid data type added to VectorState.")
        return new

    def minus(self, x: Union[np.ndarray, "VectorState"]) -> "VectorState":
        new = self.copy()
        if isinstance(x, np.ndarray):
            if x.size == self.dof:
                new.value = new.value.ravel() - x.ravel()
            else:
                raise ValueError("Mismatched state size subtracted to VectorState.")
        elif isinstance(x, VectorState):
            if x.dof == self.dof:
                new.value = new.value.ravel() - x.value.ravel()
            else:
                raise ValueError("Mismatched state size subtracted to VectorState.")
        else:
            raise ValueError("Invalid data type subtracted to VectorState.")
        return new

    def dot(self, x: "VectorState") -> "VectorState":
        new = self.copy()
        new.value = new.value.ravel().dot(x.value.ravel())
        return new

    def copy(self) -> "VectorState":
        return VectorState(self.value.copy(), self.stamp, self.id)

    @staticmethod
    def random(dof: int = 3, stamp: float = None, id: str = None) -> "VectorState":
        return VectorState(np.random.randn(dof), stamp, id)

    def inverse(self) -> "VectorState":
        return VectorState(-self.value, self.stamp, self.id)

    def plus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        return np.identity(self.dof)


class MatrixLieGroupState(State):
    """
    The MatrixLieGroupState abstract class must be inherited by subclasses to be used.
    """

    __slots__ = ["group", "direction"]

    def __init__(
        self,
        value: np.ndarray,
        group: mlg.MatrixLieGroup,
        stamp: float = None,
        id: str = None,
        direction: str = "right",
    ):
        """
        Parameters
        ----------
        value : np.ndarray
            Value of the state in the matrix form
        group : pymlg.MatrixLieGroup
            A `pymlg.MatrixLieGroup` class, such as `pymlg.SE2` or `pymlg.SO3`
        stamp : float, optional
            Timestamp, by default None
        id : str, optional
            State id for description, by default None
        direction : str, optional
            Uncertainty direction either "left" or "right", by default "right"
            Defines :math:`\\delta \\mathbf{x}` as either

            .. math::

                \\mathbf{X} = \\mathbf{X} \\exp(\\delta \\mathbf{x}^\\wedge) \\text{ (right)}
                \\mathbf{X} = \\exp(\\delta \\mathbf{x}^\\wedge) \\mathbf{X} \\text{ (left)}
        """
        value = np.array(value, dtype=np.float64)

        if value.size == group.dof:
            value = group.Exp(value)
        elif value.shape[0] != value.shape[1]:
            raise ValueError(
                f"value must either be a {group.dof}-length vector of exponential"
                "coordinates or a matrix direct element of the group."
            )

        self.direction = direction
        self.group = group
        super(MatrixLieGroupState, self).__init__(value, self.group.dof, stamp, id)
        self.value: np.ndarray = self.value  # just for type hinting

    def __repr__(self):
        value_str = str(self.value).split("\n")
        value_str = "\n".join(["    " + s for s in value_str])
        s = [
            f"{self.__class__.__name__}(stamp={self.stamp},"
            + f" id={self.id}, direction={self.direction})",
            f"{value_str}",
        ]
        return "\n".join(s)

    @property
    def pose(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have pose property".format(self.__class__.__name__)
        )

    @property
    def attitude(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have attitude property".format(self.__class__.__name__)
        )

    @property
    def position(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have position property".format(self.__class__.__name__)
        )

    @property
    def velocity(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have velocity property".format(self.__class__.__name__)
        )

    def get_value(self) -> np.ndarray:
        return self.group.Log(self.value).ravel()

    def plus(
        self, x: Union[np.ndarray, "MatrixLieGroupState"]
    ) -> "MatrixLieGroupState":
        new = self.copy()
        if isinstance(x, np.ndarray):
            if x.size == self.dof:
                if self.direction == "right":
                    new.value = self.value @ self.group.Exp(x)
                elif self.direction == "left":
                    new.value = self.group.Exp(x) @ self.value
                else:
                    raise ValueError("direction must either be 'left' or 'right'.")
            else:
                raise ValueError("Mismatched state size added to MatrixLieGroupState.")
        elif isinstance(x, MatrixLieGroupState):
            if x.dof == self.dof:
                if self.direction == "right":
                    new.value = self.value @ x.value
                elif self.direction == "left":
                    new.value = x.value @ self.value
                else:
                    raise ValueError("direction must either be 'left' or 'right'.")
            else:
                raise ValueError("Mismatched state size added to MatrixLieGroupState.")
        else:
            raise ValueError("Invalid data type added to MatrixLieGroupState.")
        return new

    def minus(
        self, x: Union[np.ndarray, "MatrixLieGroupState"]
    ) -> "MatrixLieGroupState":
        new = self.copy()
        if isinstance(x, np.ndarray):
            if x.size == self.dof:
                if self.direction == "right":
                    new.value = self.group.inverse(x) @ self.value
                elif self.direction == "left":
                    new.value = self.value @ self.group.inverse(x)
                else:
                    raise ValueError("direction must either be 'left' or 'right'.")
            else:
                raise ValueError(
                    "Mismatched state size subtracted to MatrixLieGroupState."
                )
        elif isinstance(x, MatrixLieGroupState):
            if x.dof == self.dof:
                if self.direction == "right":
                    new.value = self.group.inverse(x.value) @ self.value
                elif self.direction == "left":
                    new.value = self.value @ self.group.inverse(x.value)
                else:
                    raise ValueError("direction must either be 'left' or 'right'.")
            else:
                raise ValueError(
                    "Mismatched array size subtracted to MatrixLieGroupState."
                )
        else:
            raise ValueError("Invalid data type subtracted to MatrixLieGroupState.")
        return new

    def dot(self, other: "MatrixLieGroupState") -> "MatrixLieGroupState":
        """
        Matrix multiplication of two MatrixLieGroupStates.

        Parameters
        ----------
        other : MatrixLieGroupState
            Another MatrixLieGroupState object

        Returns
        -------
        MatrixLieGroupState
            The result of the matrix multiplication
        """
        new = self.copy()
        new.value = self.value @ other.value
        return new

    def copy(self) -> "MatrixLieGroupState":
        # Check if instance of this class as opposed to a child class
        if type(self) == MatrixLieGroupState:
            return MatrixLieGroupState(
                self.value.copy(),
                self.group,
                self.stamp,
                self.id,
                self.direction,
            )
        else:
            return self.__class__(
                self.value.copy(),
                self.stamp,
                self.id,
                self.direction,
            )

    def inverse(self) -> "MatrixLieGroupState":
        new = self.copy()
        new.value = self.group.inverse(self.value)
        return new

    def plus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        if self.direction == "right":
            jac = self.group.right_jacobian(dx)
        elif self.direction == "left":
            jac = self.group.left_jacobian(dx)
        else:
            raise ValueError("direction must either be 'left' or 'right'.")
        return jac

    def minus_jacobian(self, x: "MatrixLieGroupState") -> np.ndarray:
        dx = self.minus(x).get_value()
        if self.direction == "right":
            jac = self.group.right_jacobian_inv(dx)
        elif self.direction == "left":
            jac = self.group.left_jacobian_inv(dx)
        else:
            raise ValueError("direction must either be 'left' or 'right'.")
        return jac

    @staticmethod
    @abstractmethod
    def random():
        """
        Returns an instance of this State with random values.
        """
        pass

    @staticmethod
    @abstractmethod
    def identity():
        """
        Returns an instance of this State with identity values.
        """
        pass

    @staticmethod
    @abstractmethod
    def jacobian_from_blocks(self, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have jacobian_from_blocks method".format(
                self.__class__.__name__
            )
        )


class SO3State(MatrixLieGroupState):
    group = mlg.SO3

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        id: str = None,
        direction="right",
    ):
        super().__init__(value, self.group, stamp, id, direction)

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0:3, 0:3]

    @attitude.setter
    def attitude(self, C: np.ndarray):
        self.value[0:3, 0:3] = C

    @staticmethod
    def random(stamp: float = None, id: str = None, direction="right"):
        return SO3State(mlg.SO3.random(), stamp=stamp, id=id, direction=direction)

    @staticmethod
    def identity(stamp: float = None, id: str = None, direction="right"):
        return SO3State(mlg.SO3.identity(), stamp=stamp, id=id, direction=direction)

    @staticmethod
    def jacobian_from_blocks(
        attitude: np.ndarray = None,
    ):
        for jac in [attitude]:
            if jac is not None:
                dim = jac.shape[0]

        if attitude is None:
            attitude = np.zeros((dim, 3))

        return attitude


class SE3State(MatrixLieGroupState):
    group = mlg.SE3

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        id: str = None,
        direction="right",
    ):
        super().__init__(value, self.group, stamp, id, direction)

    @property
    def pose(self) -> np.ndarray:
        return self.value

    @pose.setter
    def pose(self, T):
        self.value = T

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0:3, 0:3]

    @attitude.setter
    def attitude(self, C: np.ndarray):
        self.value[0:3, 0:3] = C

    @property
    def position(self) -> np.ndarray:
        return self.value[0:3, 3]

    @position.setter
    def position(self, r: np.ndarray):
        self.value[0:3, 3] = r.ravel()

    @staticmethod
    def random(stamp: float = None, id: str = None, direction="right"):
        return SE3State(mlg.SE3.random(), stamp=stamp, id=id, direction=direction)

    @staticmethod
    def identity(stamp: float = None, id: str = None, direction="right"):
        return SE3State(mlg.SE3.identity(), stamp=stamp, id=id, direction=direction)

    @staticmethod
    def jacobian_from_blocks(
        attitude: np.ndarray = None,
        position: np.ndarray = None,
    ):
        for jac in [attitude, position]:
            if jac is not None:
                dim = jac.shape[0]

        if attitude is None:
            attitude = np.zeros((dim, 3))
        if position is None:
            position = np.zeros((dim, 3))

        return np.block([attitude, position])


class SE23State(MatrixLieGroupState):
    group = mlg.SE23

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        id: str = None,
        direction="right",
    ):
        super().__init__(value, self.group, stamp, id, direction)

    @property
    def pose(self) -> np.ndarray:
        return self.value[0:5, 0:5]

    @pose.setter
    def pose(self, T):
        self.value[0:5, 0:5] = T

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0:3, 0:3]

    @attitude.setter
    def attitude(self, C: np.ndarray):
        self.value[0:3, 0:3] = C

    @property
    def velocity(self) -> np.ndarray:
        return self.value[0:3, 3]

    @velocity.setter
    def velocity(self, v: np.ndarray):
        self.value[0:3, 3] = v.ravel()

    @property
    def position(self) -> np.ndarray:
        return self.value[0:3, 4]

    @position.setter
    def position(self, r: np.ndarray):
        self.value[0:3, 4] = r.ravel()

    @staticmethod
    def random(stamp: float = None, id: str = None, direction="right"):
        return SE23State(mlg.SE23.random(), stamp=stamp, id=id, direction=direction)

    @staticmethod
    def identity(stamp: float = None, id: str = None, direction="right"):
        return SE23State(mlg.SE23.identity(), stamp=stamp, id=id, direction=direction)

    @staticmethod
    def jacobian_from_blocks(
        attitude: np.ndarray = None,
        velocity: np.ndarray = None,
        position: np.ndarray = None,
    ):
        for jac in [attitude, velocity, position]:
            if jac is not None:
                dim = jac.shape[0]

        if attitude is None:
            attitude = np.zeros((dim, 3))
        if velocity is None:
            velocity = np.zeros((dim, 3))
        if position is None:
            position = np.zeros((dim, 3))

        return np.block([attitude, velocity, position])


class CompositeState(State):
    """
    CompositeState intended to hold a list of State objects.
    It is possible to access sub-states in the composite states both by index and by ID.
    """

    def __init__(self, state_list: list[State], stamp: float = None, id: str = None):
        """
        Parameters
        ----------
        state_list : list[State]
            List of State that forms this composite state
        stamp : float, optional
            Timestamp, by default None
        id : str, optional
            State id for description, by default None
        """
        self.value = state_list
        self.stamp = stamp
        self.id = id

    def __getstate__(self):
        """
        Get the state of the object for pickling.
        """
        # When using __slots__ the pickle module expects a tuple from __getstate__.
        # See https://stackoverflow.com/questions/1939058/simple-example-of-use-of-setstate-and-getstate/41754104#41754104
        return (
            None,
            {
                "value": self.value,
                "stamp": self.stamp,
                "id": self.id,
            },
        )

    def __setstate__(self, attributes):
        """
        Set the state of the object for unpickling.
        """
        # When using __slots__ the pickle module sends a tuple for __setstate__.
        # See https://stackoverflow.com/questions/1939058/simple-example-of-use-of-setstate-and-getstate/41754104#41754104

        attributes = attributes[1]
        self.value = attributes["value"]
        self.stamp = attributes["stamp"]
        self.id = attributes["id"]

    def __repr__(self):
        substate_line_list = []
        for v in self.value:
            substate_line_list.extend(v.__repr__().split("\n"))
        substates_str = "\n".join(["    " + s for s in substate_line_list])
        s = [
            f"{self.__class__.__name__}(stamp={self.stamp}, id={self.id}) with substates:",
            substates_str,
        ]
        return "\n".join(s)

    @property
    def dof(self):
        return sum([x.dof for x in self.value])

    def add_state(self, state: State):
        """
        Adds a state and it's corresponding slice to the composite state.
        """
        self.value.append(state)

    def get_slices(self) -> list[slice]:
        """
        Get slices for each state in the list of states.
        """
        slices = []
        counter = 0
        for state in self.value:
            slices.append(slice(counter, counter + state.dof))
            counter += state.dof

        return slices

    def get_slice_by_id(self, id: str, slices=None):
        """
        Get slice of a particular id in the list of states.
        """

        if slices is None:
            slices = self.get_slices()

        idx = self.get_index_by_id(id)
        return slices[idx]

    def get_index_by_id(self, id: str):
        """
        Get index of a particular id in the list of states.
        """
        return [x.id for x in self.value].index(id)

    def set_state_by_id(self, state: State, id: str):
        """
        Set the whole sub-state by ID.
        """
        idx = self.get_index_by_id(id)
        self.value[idx] = state

    def get_state_by_id(self, id: str) -> State:
        """
        Get state object by ID.
        """
        idx = self.get_index_by_id(id)
        return self.value[idx]

    def remove_state_by_id(self, id: str):
        """
        Removes a given state by ID.
        """
        idx = self.get_index_by_id(id)
        self.value.pop(idx)

    def set_stamp_by_id(self, stamp: float, id):
        """
        Set the timestamp of a sub-state by id.
        """
        idx = self.get_index_by_id(id)
        self.value[idx].stamp = stamp

    def set_stamp_for_all(self, stamp: float):
        """
        Set the timestamp of all substates.
        """
        for state in self.value:
            state.stamp = stamp

    def to_list(self):
        """
        Converts the CompositeState object back into a list of states.
        """
        return self.value

    def plus_by_id(
        self, x: State, id: str, new_stamp: float = None
    ) -> "CompositeState":
        """
        Add a specific sub-state.
        """
        new = self.copy()
        idx = new.get_index_by_id(id)
        new.value[idx].plus(x)
        if new_stamp is not None:
            new.set_stamp_by_id(new_stamp, id)

        return new

    def minus_by_id(
        self, x: State, id: str, new_stamp: float = None
    ) -> "CompositeState":
        """
        Add a specific sub-state.
        """
        new = self.copy()
        idx = new.get_index_by_id(id)
        new.value[idx].minus(x)
        if new_stamp is not None:
            new.set_stamp_by_id(new_stamp, id)

        return new

    def get_value(self) -> np.ndarray:
        idx = 0
        value = np.zeros(self.dof)
        for state in self.value:
            value[idx : idx + state.dof] = state.get_value()
            idx += state.dof
        return value

    def plus(
        self, x: Union[np.ndarray, "CompositeState"], new_stamp: float = None
    ) -> "CompositeState":
        new = self.copy()
        if isinstance(x, np.ndarray):
            for i, state in enumerate(new.value):
                new.value[i] = state.plus(x[: state.dof])
                x = x[state.dof :]
        elif isinstance(x, CompositeState):
            for i, state in enumerate(new.value):
                new.value[i] = state.plus(x.value[i])

        if new_stamp is not None:
            new.set_stamp_for_all(new_stamp)

        return new

    def minus(
        self, x: Union[np.ndarray, "CompositeState"], new_stamp: float = None
    ) -> "CompositeState":
        new = self.copy()
        if isinstance(x, np.ndarray):
            for i, state in enumerate(new.value):
                new.value[i] = state.minus(x[: state.dof])
                x = x[state.dof :]
        elif isinstance(x, CompositeState):
            for i, state in enumerate(new.value):
                new.value[i] = state.minus(x.value[i])

        if new_stamp is not None:
            new.set_stamp_for_all(new_stamp)

        return new

    def copy(self) -> "CompositeState":
        return self.__class__(
            [state.copy() for state in self.value], self.stamp, self.id
        )

    def random(self) -> "CompositeState":
        raise NotImplementedError(
            "{0} does not have random method".format(self.__class__.__name__)
        )

    def inverse(self) -> "CompositeState":
        new = self.copy()
        for i, state in enumerate(self.value):
            new.value[i] = state.inverse()
        return new

    def dot(self, x: "CompositeState") -> "CompositeState":
        new = self.copy()
        for i, state in enumerate(new.value):
            new.value[i] = state.dot(x.value[i])
        return new

    def jacobian_from_blocks(self, block_dict: dict):
        """
        Returns the jacobian of the entire composite state given jacobians
        associated with some of the substates. These are provided as a dictionary
        with the the keys being the substate IDs.
        """
        block: np.ndarray = list(block_dict.values())[0]
        m = block.shape[0]
        jac = np.zeros((m, self.dof))
        slices = self.get_slices()
        for id, block in block_dict.items():
            slc = self.get_slice_by_id(id, slices)
            jac[:, slc] = block

        return jac
