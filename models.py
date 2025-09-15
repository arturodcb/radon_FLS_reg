
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dataTypes
    import states


class Increment(ABC):

    __slots__ = [
        "dof",
        "stamps",
        "input_covariance",
        "id",
        "direction",
        "gravity",
        "value",
        "bias",
        "jacobian",
        "covariance",
        "method",
    ]

    def __repr__(self):
        return f"{self.__class__.__name__} at {hex(id(self))}"

    @abstractmethod
    def increment(self, u: "dataTypes.Measurement", dt: float):
        pass

    @abstractmethod
    def plus(self, w: np.ndarray) -> "Increment":
        pass

    @abstractmethod
    def copy(self) -> "Increment":
        pass

    @abstractmethod
    def new(self) -> "Increment":
        pass

    @abstractmethod
    def reset(self, x: "states.State"):
        pass


class ProcessModel(ABC):
    """
    Abstract process model base class for process models of the form

    .. math::
        \\mathcal{X}_k = f(\\mathcal{X}_{k-1}, \\mathbf{u}_{k-1}, \\Delta t),

    where :math:`\\mathbf{u}_{k-1}` is the input, :math:`\\mathbf{w}_{k-1} \\sim
    \\mathcal{N}(\\mathbf{0}, \\mathbf{Q}_{k-1})` is additive Gaussian noise, and
    :math:`\\Delta t` is the time period between the two states.

    To define a process model, you must inherit from this class and implement
    the ``evaluate`` method and specify the input covariance matrix
    :math:`\\mathbf{Q}`.

    With the input covariance matrix, the covariance of the process model error
    is approximated using a linearization procedure,

    .. math::

        \\mathbf{Q}_{k-1} = \\mathbf{L}_{k-1} \\mathbf{Q} \\Delta t \\mathbf{L}_{k-1}^T,

    where :math:`\\mathbf{L}_{k-1} = D \\mathbf{f}(\\mathcal{X}_{k-1}, \\mathbf{u}_{k-1}, dt) /
    D \\mathbf{u}_{k-1}` is the *input jacobian*. This is calculated using finite
    difference by default, but can be overridden by implementing the
    ``input_jacobian`` method.
    """

    __slots__ = ["_input_covariance"]

    def __repr__(self):
        return f"{self.__class__.__name__} at {hex(id(self))}"

    @abstractmethod
    def evaluate(
        self, x: "states.State", u: "dataTypes.Measurement", dt: float
    ) -> "states.State":
        """
        Implementation of

        .. math::
            \\mathbf{f}(\\mathcal{X}_{k-1}, \\mathbf{u}_{k-1}, \\Delta t).

        Parameters
        ----------
        x : states.State
            State at time :math:`k-1`
        u : dataTypes.Measurement
            The input measurement :math:`\\mathbf{u}_{k-1}`
            The actual numerical value is accessed via `u.value`
        dt : float
            The time interval :math:`\\Delta t` between the two states

        Returns
        -------
        states.State
            State at time :math:`k`
        """
        pass

    def jacobian(
        self, x: "states.State", u: "dataTypes.Measurement", dt: float
    ) -> np.ndarray:
        """
        Implementation of the process model Jacobian with respect to the state,

        .. math::
            \\mathbf{F}_{k-1} =
            \\frac{D \\mathbf{f}(\\mathcal{X}_{k-1}, \\mathbf{u}_{k-1}, \\Delta t)}
            {D \\mathcal{X}_{k-1}}.

        This is calculated using finite difference by default, but can be
        overridden by implementing this method.

        Parameters
        ----------
        x : states.State
            State at time :math:`k-1`
        u : dataTypes.Measurement
            The input measurement :math:`\\mathbf{u}_{k-1}`
            The actual numerical value is accessed via `u.value`
        dt : float
            The time interval :math:`\\Delta t` between the two states

        Returns
        -------
        np.ndarray
            Process model Jacobian with respect to the state :math:`\\mathbf{F}_{k-1}`
        """
        return self._state_jacobian_fd(x, u, dt)

    def evaluate_with_jacobian(
        self, x: "states.State", u: "dataTypes.Measurement", dt: float
    ) -> tuple["states.State", np.ndarray]:
        """
        Evaluates the process model and simultaneously returns the Jacobian.
        This is useful to override for performance reasons when
        the model evaluation and Jacobian have a lot of common calculations,
        and it is more efficient to calculate them in the same function call.

        Parameters
        ----------
        x : states.State
            State at time :math:`k-1`
        u : dataTypes.Measurement
            The input measurement :math:`\\mathbf{u}_{k-1}`
            The actual numerical value is accessed via `u.value`
        dt : float
            The time interval :math:`\\Delta t` between the two states

        Returns
        -------
        tuple[states.State, np.ndarray]
            State at time :math:`k` and the process model Jacobian
        """
        return self.evaluate(x, u, dt), self.jacobian(x, u, dt)

    def covariance(
        self, x: "states.State", u: "dataTypes.Measurement", dt: float
    ) -> np.ndarray:
        """
        Covariance matrix :math:`\\mathbf{Q}_{k-1}` of the additive Gaussian noise
        :math:`\\mathbf{w}_{k-1} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{Q}_{k-1})`.
        This is calculated using

        .. math::

            \\mathbf{Q}_{k-1} = \\mathbf{L}_{k-1} \\mathbf{Q} \\Delta t \\mathbf{L}_{k-1}^T,

        where the *input jacobian* is computed using finite difference by default.
        This can be overridden by implementing this method.

        Parameters
        ----------
        x : states.State
            State at time :math:`k-1`
        u : dataTypes.Measurement
            The input measurement :math:`\\mathbf{u}_{k-1}`
            The actual numerical value is accessed via `u.value`
        dt : float
            The time interval :math:`\\Delta t` between the two states

        Returns
        -------
        np.ndarray
            Covariance matrix :math:`\\mathbf{Q}_{k-1}`
        """
        L = np.atleast_2d(self._input_jacobian_fd(x, u, dt))
        Q = np.atleast_2d(self._input_covariance(x, u, dt))
        return L @ (Q * dt) @ L.T

    def sqrt_information(
        self, x: "states.State", u: "dataTypes.Measurement", dt: float
    ) -> np.ndarray:
        """
        Returns the cholesky decomposition of the information matrix, :math:`\\mathbf{L}`.
        This is useful in nonlinear optimization where the information matrix is
        premultiplied to the errors.

        Parameters
        ----------
        x : states.State
            State at time :math:`k-1`
        u : dataTypes.Measurement
            The input measurement :math:`\\mathbf{u}_{k-1}`
            The actual numerical value is accessed via `u.value`
        dt : float
            The time interval :math:`\\Delta t` between the two states

        Returns
        -------
        np.ndarray
            Cholesky decomposition of the information matrix
        """
        Qd = np.atleast_2d(self.covariance(x, u, dt))
        return np.linalg.cholesky((np.linalg.inv(Qd) + np.linalg.inv(Qd).T) / 2)

    def _state_jacobian_fd(
        self,
        x: "states.State",
        u: "dataTypes.Measurement",
        dt: float,
        step_size=1e-6,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Calculates the state jacobian with finite difference.

        Parameters
        ----------
        x : states.State
            State at time :math:`k-1`
        u : dataTypes.Measurement
            The input measurement :math:`\\mathbf{u}_{k-1}`
            The actual numerical value is accessed via `u.value`
        dt : float
            The time interval :math:`\\Delta t` between the two states
        step_size : float, optional
            Finite difference step size, by default 1e-6

        Returns
        -------
        np.ndarray
            Discrete Jacobian matrix with respect to state
        """
        Y_bar = self.evaluate(x.copy(), u, dt, *args, **kwargs)
        jac_fd = np.zeros((x.dof, x.dof))
        for i in range(x.dof):
            dx = np.zeros((x.dof))
            dx[i] = step_size
            x_pert = x.plus(dx)
            Y = self.evaluate(x_pert, u, dt, *args, **kwargs)
            jac_fd[:, i] = Y.minus(Y_bar).get_value() / step_size

        return jac_fd

    def _input_jacobian_fd(
        self,
        x: "states.State",
        u: "dataTypes.Measurement",
        dt: float,
        step_size=1e-6,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Calculates the input jacobian with finite difference.

        Parameters
        ----------
        x : states.State
            State at time :math:`k-1`
        u : dataTypes.Measurement
            The input measurement :math:`\\mathbf{u}_{k-1}`
            The actual numerical value is accessed via `u.value`
        dt : float
            The time interval :math:`\\Delta t` between the two states
        step_size : float, optional
            Finite difference step size, by default 1e-6

        Returns
        -------
        np.ndarray
            Discrete Jacobian matrix with respect to input
        """
        Y_bar = self.evaluate(x.copy(), u.copy(), dt, *args, **kwargs)
        jac_fd = np.zeros((x.dof, u.dof))
        for i in range(u.dof):
            du = np.zeros((u.dof,))
            du[i] = step_size
            Y = self.evaluate(x.copy(), u.plus(du), dt, *args, **kwargs)
            jac_fd[:, i] = Y.minus(Y_bar).get_value() / step_size

        return jac_fd


class MeasurementModel(ABC):
    """
    Abstract measurement model base class, used to implement measurement models of the form

    .. math::
        \\mathbf{y}_{k} = \\mathbf{g}(\\mathbf{x}_{k}) + \\mathbf{v}_{k},

    where :math:`\\mathbf{v}_{k} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{R}_{k})`.
    """

    __slots__ = ["freq", "type", "_covariance"]

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def evaluate(self, x: "states.State") -> np.ndarray:
        """
        Implementation of

        .. math::
            \\mathbf{g}(\\mathbf{x}_{k}).

        Parameters
        ----------
        x : state.State
            State at time :math:`k`

        Returns
        -------
        np.ndarray
            Expected measurement at time :math`k`
        """
        pass

    def jacobian(self, x: "states.State") -> np.ndarray:
        """
        Implementation of the measurement model Jacobian with respect to the state,

        .. math::
            \\mathbf{G} = \\partial \\mathbf{g}(\\mathbf{x})/ \\partial \\mathbf{x}.

        This is calculated using finite difference by default, but can be
        overridden by implementing this method.

        Parameters
        ----------
        x : states.State
            State at time :math:`k`

        Returns
        -------
        np.ndarray
            Measurement model Jacobian with respect to the state :math:`mathbf{G}_{k}`
        """
        return self._state_jacobian_fd(x)

    def evaluate_with_jacobian(
        self, x: "states.State"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the process model and simultaneously returns the Jacobian.
        This is useful to override for performance reasons when
        the model evaluation and Jacobian have a lot of common calculations,
        and it is more efficient to calculate them in the same function call.

        Parameters
        ----------
        x : states.State
            State at time :math:`k`

        Returns
        -------
        tuple[states.State, np.ndarray]
            State at time :math:`k` and the process model Jacobian
        """
        return self.evaluate(x), self.jacobian(x)

    def covariance(self, x: "states.State") -> np.ndarray:
        """
        Covariance matrix :math:`\\mathbf{R}_{k}` of the additive Gaussian noise
        :math:`\\mathbf{w}_{k} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{R}_{k})`.
        This is calculated using

        .. math::
            \\mathbf{R}_{k} = \\mathbf{M}_{k} \\mathbf{R} \\mathbf{M}_{k},

        where the *measurement jacobian* :math:`\\mathbf{M}_{k}` must be computed.

        Parameters
        ----------
        x : states.State
            State at time :math:`k`

        Returns
        -------
        np.ndarray
            Covariance matrix :math:`\\mathbf{R}_{k}`
        """
        pass

    def sqrt_information(self, x: "states.State") -> np.ndarray:
        """
        Returns the cholesky decomposition of the information matrix, :math:`\\mathbf{L}`.
        This is useful in nonlinear optimization where the information matrix is
        premultiplied to the errors.

        Parameters
        ----------
        x : states.State
            State at time :math:`k-1`

        Returns
        -------
        np.ndarray
            Cholesky decomposition of the information matrix
        """
        R = np.atleast_2d(self.covariance(x))
        return np.linalg.cholesky(np.linalg.inv(R))

    def _state_jacobian_fd(self, x: "states.State", step_size=1e-6) -> np.ndarray:
        """
        Calculates the model jacobian with finite difference.

        Parameters
        ----------
        x : states.State
            Current state
        step_size : float, optional
            Finite difference step size, by default 1e-6

        Returns
        -------
        np.ndarray
            Discrete Jacobian matrix
        """
        N = x.dof
        y = self.evaluate(x)
        m = y.size
        jac_fd = np.zeros((m, N))
        for i in range(N):
            dx = np.zeros((N, 1))
            dx[i, 0] = step_size
            x_temp = x.plus(dx)
            jac_fd[:, i] = (self.evaluate(x_temp) - y).flatten() / step_size

        return jac_fd


class BodyFrameVelocity(ProcessModel):
    """
    The body-frame velocity process model assumes that the input contains
    both translational and angular velocity measurements, both relative to
    a local reference frame, but resolved in the robot body frame.

    .. math::
        \mathbf{T}_k = \mathbf{T}_{k-1} \exp(\Delta t \mathbf{u}_{k-1}^\wedge)

    This is commonly the process model associated with SE(n).

    This class is comptabile with ``SO2State, SO3State, SE2State, SE3State, SE23State``.
    """

    def __init__(self, Q: np.ndarray):
        self._Q = Q

    def evaluate(
        self, x: "states.MatrixLieGroupState", u: np.ndarray, dt: float
    ) -> "states.MatrixLieGroupState":
        """
        Implementation of
        .. math::
            \\mathbf{T}_k = \\mathbf{T}_{k-1} \\exp(\\Delta t \\mathbf{u}_{k-1}^\\wedge)

        Parameters
        ----------
        x : states.MatrixLieGroupState
            State at time :math:`k-1`
        u : np.ndarray
            The input measurement :math:`\\mathbf{u}_{k-1}`
            The actual numerical value is accessed via `u.value`
        dt : float
            The time interval :math:`\\Delta t` between the two states

        Returns
        -------
        states.MatrixLieGroupState
            State at time :math:`k`
        """
        x = x.copy()
        x.value = x.value @ x.group.Exp(u * dt)
        return x

    def jacobian(
        self, x: "states.MatrixLieGroupState", u: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Implementation of the process model Jacobian with respect to the state,

        .. math::
            \\mathbf{F}_{k-1} =
            \\frac{D \\mathbf{f}(\\mathcal{X}_{k-1}, \\mathbf{u}_{k-1}, \\Delta t)}
            {D \\mathcal{X}_{k-1}}.
        This is calculated here.

        Parameters
        ----------
        x : states.MatrixLieGroupState
            State at time :math:`k-1`
        u : np.ndarray
            The input measurement :math:`\\mathbf{u}_{k-1}`
            The actual numerical value is accessed via `u.value`
        dt : float
            The time interval :math:`\\Delta t` between the two states

        Returns
        -------
        np.ndarray
            Process model Jacobian with respect to the state :math:`\\mathbf{F}_{k-1}`
        """
        if x.direction == "right":
            return x.group.adjoint(x.group.Exp(-u * dt))
        elif x.direction == "left":
            return np.identity(x.dof)

    def covariance(
        self, x: "states.MatrixLieGroupState", u: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Covariance matrix :math:`\\mathbf{Q}_{k-1}` of the additive Gaussian noise
        :math:`\\mathbf{w}_{k-1} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{Q}_{k-1})`.
        This is calculated using
        .. math::
            \\mathbf{Q}_{k-1} = \\mathbf{L}_{k-1} \\mathbf{Q} \\Delta t \\mathbf{L}_{k-1}^T,

        where the *input jacobian* `\\mathbf{L}_{k-1}` is computed here.

        Parameters
        ----------
        x : states.MatrixLieGroupState
            State at time :math:`k-1`
        u : np.ndarray
            The input measurement :math:`\\mathbf{u}_{k-1}`
            The actual numerical value is accessed via `u.value`
        dt : float
            The time interval :math:`\\Delta t` between the two states

        Returns
        -------
        np.ndarray
            Covariance matrix :math:`\\mathbf{Q}_{k-1}`
        """
        if x.direction == "right":
            L = dt * x.group.left_jacobian(-u * dt)
        elif x.direction == "left":
            Ad = x.group.adjoint(x.value @ x.group.Exp(u * dt))
            L = dt * Ad @ x.group.left_jacobian(-u * dt)

        return L @ self._Q @ L.T
