# python imports
import numpy as np
from comet.state.state_conversions import cartesian_to_elements

# COMET imports
from comet.utilities.constants import Constants as c


class State:
    """Class that represents a cartesian state.

    Example Constructions:
        * state = State(x, y, z, vx, vy, vz)
        * state = State(pos, vel)
        * state = State([x, y, z, vx, vy, vz])
    """

    def __init__(self, *args):
        """Defines a Cartesian State vector.

        Construct from 6 state components in km and km/s.

        Args:
            x (int|float): X Component of Position.
            y (int|float): Y Component of Position.
            z (int|float): Z Component of Position.
            Vx (int|float): X Component of Velocity.
            Vy (int|float): Y Component of Velocity.
            Vz (int|float): Z Component of Velocity.

        Construct from a Position and Velocity array in km and km/s.

        Args:
            position (list|np.ndarray): Position array.
            velocity (list|np.ndarray): Velocity array.

        Construct from a state array in km and km/s.

        Args:
            state (list|np.ndarray): State array.
        """
        # Construct Class
        if len(args) == 6:
            # 6 individual components for Position and Velocity
            if not np.all([isinstance(arg, int | float | np.int64 | np.float64) for arg in args]):
                raise TypeError(
                    "State(): values must be an integer or float when providing 6 inputs"
                )
            self._raw = np.array(args)

        elif len(args) == 2:
            # Position and Velocity vectors
            if not np.all([isinstance(arg, list | np.ndarray) for arg in args]):
                raise TypeError("State(): Position and Velocity must be Arrays of length 3")
            for arg in args:
                if not np.all([isinstance(a, int | float | np.int64 | np.float64) for a in arg]):
                    raise TypeError(
                        "State(): Values inside position and velocity must be an integer or float"
                    )
            self._raw = np.append(args[0], args[1], axis=-1)

        elif len(args) == 1:
            # Full State Vector
            if not isinstance(args[0], list | np.ndarray):
                raise TypeError("State(): State must be an Array of length 6")
            if len(args[0]) != 6:
                raise TypeError("State(): State must be an Array of length 6")
            if not np.all(
                [isinstance(arg, int | float | np.int64 | np.float64) for arg in args[0]]
            ):
                raise TypeError("State(): Values inside State array must be an integer or float")
            self._raw = np.array(args[0])

        else:
            raise ValueError("State(): Invalid number of inputs")

    def copy(self):
        """Returns a copy of the State.

        Returns:
            state: Copy of the State.
        """
        return State(self._raw)

    def position(self) -> np.ndarray:
        """Returns the position of the State.

        Returns:
            position (np.ndarray): State position in km.
        """
        return self._raw[:3]

    def velocity(self) -> np.ndarray:
        """Returns the velocity of the State.

        Returns:
            velocity (np.ndarray): State velocity in km/s.
        """
        return self._raw[3:]

    @property
    def x(self) -> float:
        """X component of position in km."""
        return self._raw[0]

    @property
    def y(self) -> float:
        """Y component of position in km."""
        return self._raw[1]

    @property
    def z(self) -> float:
        """Z component of position in km."""
        return self._raw[2]

    @property
    def vx(self) -> float:
        """X component of velocity in km/s."""
        return self._raw[3]

    @property
    def vy(self) -> float:
        """Y component of velocity in km/s."""
        return self._raw[4]

    @property
    def vz(self) -> float:
        """Z component of velocity in km/s."""
        return self._raw[5]

    def position_magnitude(self) -> float:
        """Returns the magnitude of the position vector.

        Returns:
            float: Position magnitude in km.
        """
        return np.linalg.norm(self._raw[:3])

    def velocity_magnitude(self) -> float:
        """Returns the magnitude of the velocity vector.

        Returns:
            float: Velocity magnitude in km/s.
        """
        return np.linalg.norm(self._raw[3:])

    def angular_momentum(self) -> np.ndarray:
        """Returns the specific relative angular momentum of the State.

        Returns:
            angular_momentum (np.ndarray): Specific Relative Angular Momentum in km^2/s.
        """
        return np.cross(self._raw[:3], self._raw[3:])

    def angular_momentum_magnitude(self) -> float:
        """Returns the magnitude of the specific angular momentum.

        Returns:
            float: Angular momentum magnitude in km^2/s.
        """
        return np.linalg.norm(self.angular_momentum())

    def semimajor_axis(self) -> float:
        """Returns the Semi-Major Axis of the State.

        For elliptical orbits, returns positive value.
        For hyperbolic orbits, returns negative value.
        For parabolic orbits, returns infinity.

        Returns:
            sma (float): Semi-Major Axis in km.
        """
        # Position and Velocity Magnitudes
        r = self.position_magnitude()
        v = self.velocity_magnitude()

        return 1.0 / ((2.0 / r) - (v**2) / c.MU_EARTH)

    def eccentricity(self) -> float:
        """Returns the Eccentricity of the State.

        Returns:
            ecc (float): Eccentricity (0 = circular, <1 = elliptical, 1 = parabolic, >1 = hyperbolic).
        """
        # Angular Momentum and Semi-Major Axis
        h = self.angular_momentum_magnitude()
        a = self.semimajor_axis()

        # Handle hyperbolic orbits (negative a)
        parameter = (h**2) / (c.MU_EARTH * np.abs(a))
        if a > 0:
            # Elliptical orbit
            # Clamp parameter to handle numerical errors near circular orbits
            if parameter >= 1.0:
                return 0.0
            return np.sqrt(1.0 - parameter)
        else:
            # Hyperbolic orbit
            return np.sqrt(1.0 + parameter)

    def inclination(self) -> float:
        """Returns the Inclination of the State.

        Returns:
            inc (float): Inclination in rad.
        """
        # Angular Momentum
        h_vec = self.angular_momentum()

        return np.arccos(h_vec[2] / np.linalg.norm(h_vec))

    def perigee(self) -> float:
        """Returns the Orbital Perigee in km.

        Returns:
            perigee (float): Orbital Perigee.
        """
        a = self.semimajor_axis()
        e = self.eccentricity()

        return a * (1 - e)

    def apogee(self) -> float:
        """Returns the Orbital Apogee in km.

        Returns:
            apogee (float): Orbital Apogee.
        """
        a = self.semimajor_axis()
        e = self.eccentricity()

        return a * (1 + e)

    def period(self) -> float:
        """Returns the Orbital Period in sec.

        Returns:
            period (float): Orbital Period.
        """
        a = self.semimajor_axis()

        return 2 * np.pi * np.sqrt((a**3) / c.MU_EARTH)

    def mean_motion(self) -> float:
        """Returns the Orbital Mean Motion in rad/s.

        Returns:
            mean_motion (float): Mean Motion.
        """
        a = self.semimajor_axis()

        return np.sqrt(c.MU_EARTH / (a**3))

    def specific_energy(self) -> float:
        """Returns the Specific Energy of the State.

        Returns:
            specific_energy (float): Specific Energy in km^2/s^2.
        """
        # Semi-Major Axis
        a = self.semimajor_axis()

        return -c.MU_EARTH / (2 * a)

    def orbit_type(self) -> str:
        """Returns the type of orbit based on eccentricity.

        Returns:
            str: Orbit type ('circular', 'elliptical', 'parabolic', 'hyperbolic').
        """
        e = self.eccentricity()
        if e < 0.01:
            return 'circular'
        elif e < 1.0:
            return 'elliptical'
        elif np.isclose(e, 1.0, rtol=1e-6):
            return 'parabolic'
        else:
            return 'hyperbolic'

    def inside_earth(self) -> bool:
        """Returns whether the State is currently inside Earth's Radius.

        Returns:
            inside (bool): Inside Earth.
        """
        return self.position_magnitude() <= c.RADIUS_EARTH

    def to_elements(self):
        """Returns the Classical Orbital Elements Representation.

        Returns:
            elements (Elements): Classical Orbital Elements.
        """
        from comet.state import Elements

        return Elements(cartesian_to_elements(self._raw))

    def to_dict(self):
        """Method that creates a dictionary of required inputs for State construction.

        Returns:
            constructor (dict): dictionary of required inputs for State construction.
        """
        return {"type": "State", "vector": self._raw.tolist()}

    @staticmethod
    def from_dict(dict):
        """Method that creates a State from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            state (State): State
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "State":
            raise ValueError("State(): Invalid construction dictionary")

        return State(dict["vector"])

    def __hash__(self):
        """Make State hashable for use in sets and as dictionary keys."""
        return hash(tuple(self._raw))

    def __eq__(self, other) -> bool:
        """Override Equality operator."""
        # Error checking
        if not isinstance(other, State | list | np.ndarray):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare states
        if isinstance(other, State):
            return np.allclose(self._raw, other._raw)
        elif isinstance(other, list | np.ndarray):
            return np.allclose(self._raw, other)

    def __ne__(self, other) -> bool:
        """Override Non-Equality operator."""
        # Error checking
        if not isinstance(other, State | list | np.ndarray):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare states
        if isinstance(other, State):
            return np.any(self._raw != other._raw)
        elif isinstance(other, list | np.ndarray):
            return np.any(self._raw != other)

    def __lt__(self, other) -> bool:
        """Override Less Than operator."""
        # Error checking
        raise NotImplementedError(f"Comparison is not defined for {type(other)}")

    def __le__(self, other) -> bool:
        """Override Less Than or Equal To operator."""
        # Error checking
        raise NotImplementedError(f"Comparison is not defined for {type(other)}")

    def __gt__(self, other) -> bool:
        """Override Greater Than operator."""
        # Error checking
        raise NotImplementedError(f"Comparison is not defined for {type(other)}")

    def __ge__(self, other) -> bool:
        """Override Greater Than or Equal To operator."""
        # Error checking
        raise NotImplementedError(f"Comparison is not defined for {type(other)}")

    def __iadd__(self, value):
        """Override += operator for adding values to States."""
        # Error checking
        if not isinstance(value, State | list | np.ndarray):
            raise NotImplementedError(
                f"Iterative Addition is not defined between States and {type(value)}"
            )

        # Add Value to State
        if isinstance(value, State):
            self._raw += value._raw
        elif isinstance(value, list | np.ndarray):
            self._raw += value

        return self

    def __add__(self, value):
        """Define Addition for adding values to States."""
        # Error checking
        if not isinstance(value, State | list | np.ndarray):
            raise NotImplementedError(f"Addition is not defined between States and {type(value)}")

        # Add Value to State
        if isinstance(value, State):
            state = self._raw + value._raw
        elif isinstance(value, list | np.ndarray):
            state = self._raw + value

        return State(state)

    def __radd__(self, value):
        """Define Reverse Addition for adding values to States."""
        # Error checking
        if not isinstance(value, State | list | np.ndarray):
            raise NotImplementedError(
                f"Reverse Addition is not defined between States and {type(value)}"
            )

        # Add State to Value
        if isinstance(value, State):
            state = self._raw + value._raw
        elif isinstance(value, list | np.ndarray):
            state = self._raw + value

        return State(state)

    def __isub__(self, value):
        """Override -= operator for subtracting values from States."""
        # Error checking
        if not isinstance(value, State | list | np.ndarray):
            raise NotImplementedError(
                f"Iterative Subtraction is not defined between States and {type(value)}"
            )

        # Subtract Value from State
        if isinstance(value, State):
            self._raw -= value._raw
        elif isinstance(value, list | np.ndarray):
            self._raw -= value

        return self

    def __sub__(self, value):
        """Define Subtraction for subtracting values from States."""
        # Error checking
        if not isinstance(value, State | list | np.ndarray):
            raise NotImplementedError(
                f"Subtraction is not defined between States and {type(value)}"
            )

        # Subtract Value from State
        if isinstance(value, State):
            state = self._raw - value._raw
        elif isinstance(value, list | np.ndarray):
            state = self._raw - value

        return State(state)

    def __rsub__(self, value):
        """Define Reverse Subtraction for subtracting values from States."""
        # Error checking
        if not isinstance(value, State | list | np.ndarray):
            raise NotImplementedError(
                f"Reverse Subtraction is not defined between States and {type(value)}"
            )

        # Subtract State from Value
        if isinstance(value, State):
            state = value._raw - self._raw
        elif isinstance(value, list | np.ndarray):
            state = value - self._raw

        return State(state)

    def __imul__(self, value):
        """Override *= operator for multiplying States by values."""
        # Error checking
        if not isinstance(value, float | int | np.int64 | np.float64):
            raise NotImplementedError(
                f"Iterative Multiplication is not defined between States and {type(value)}"
            )

        # Multiply State by value
        self._raw = self._raw * value

        return self

    def __mul__(self, value):
        """Define Multiplication for multiplying States by values."""
        # Error checking
        if not isinstance(value, float | int | np.int64 | np.float64):
            raise NotImplementedError(
                f"Multiplication is not defined between States and {type(value)}"
            )

        # Multiply State by value
        state = self._raw * value

        return State(state)

    def __rmul__(self, value):
        """Define Reverse Multiplication for multiplying States by values."""
        # Error checking
        if not isinstance(value, float | int | np.int64 | np.float64):
            raise NotImplementedError(
                f"Reverse Multiplication is not defined between States and {type(value)}"
            )

        # Multiply State by value
        state = value * self._raw

        return State(state)

    def __truediv__(self, value):
        """Define Division for dividing States by values."""
        # Error checking
        if not isinstance(value, float | int | np.int64 | np.float64):
            raise NotImplementedError(f"Division is not defined between States and {type(value)}")

        # Divide State by value
        state = np.divide(self._raw, value)

        return State(state)

    def __rtruediv__(self, value):
        """Define Reverse Division for dividing values by States."""
        # Error checking
        raise NotImplementedError(
            f"Reverse Division is not defined between States and {type(value)}"
        )

    def __getitem__(self, i):
        """Define key indexing for getting components in States."""
        return self._raw[i]

    def __setitem__(self, i, value):
        """Define key indexing for setting components in States."""
        self._raw[i] = value

    def __len__(self):
        """Define length of States."""
        return len(self._raw)

    def __str__(self):
        """String Representation of State Class"""
        pos = f"{self._raw[0]:05.3f}, {self._raw[1]:05.3f}, {self._raw[2]:05.3f} km"
        vel = f"{self._raw[3]:05.3f}, {self._raw[4]:05.3f}, {self._raw[5]:05.3f} km/s"
        return f"{pos}  {vel}"

    def __repr__(self):
        """Class Representation of State Class"""
        return f"State({self._raw})"
