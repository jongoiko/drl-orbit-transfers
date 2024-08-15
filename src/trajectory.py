import numpy as np
import pandas as pd


class Trajectory:
    def __init__(self):
        self._time = []
        self._position = []
        self._velocity = []
        self._thrust = []

    def clear(self):
        self._time = []
        self._position = []
        self._velocity = []
        self._thrust = []

    def log(self, time, position, velocity, thrust):
        self._time.append(time)
        self._position.append(np.array(position))
        self._velocity.append(np.array(velocity))
        self._thrust.append(np.array(thrust))

    @property
    def time(self):
        return np.array(self._time)

    @property
    def position(self):
        return np.array(self._position)

    @property
    def velocity(self):
        return np.array(self._velocity)

    @property
    def thrust(self):
        return np.array(self._thrust)

    def as_df(self):
        return pd.DataFrame(
            {
                "time_mjd2000": [time.mjd2000 for time in self._time],
                "position": self._position,
                "velocity": self._velocity,
                "thrust": self._thrust,
            }
        )
