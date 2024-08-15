#!/usr/bin/env python3

import os
import numpy as np
import gymnasium as gym
import random
import pykep as pk
from trajectory import Trajectory
from abc import ABC, abstractmethod
from math import radians
from gymnasium import spaces
from pathlib import Path
from utils import rsw_to_eci


class EarthOrbitEnv(gym.Env, ABC):
    metadata = {"render_modes": [], "render_fps": 4}

    EARTH_J2 = 0.001082

    def __init__(
        self,
        initial_orbit=None,
        render_mode=None,
        integrator_time_step=30.0,
        max_episode_length=None,
        isp=420.0,
        initial_mass=1000.0,
        initial_date=None,
        maneuver_duration=None,
        max_thrust=10000,
        gravity_model_file_path=os.path.join(
            os.path.dirname(__file__), "gravity/egm96.txt"
        ),
        position_normalization_factor=10_000_000,
        velocity_normalization_factor=10_000,
        trajectory_log_dir=None,
    ):
        self.initial_orbit = initial_orbit
        self.integrator_time_step = integrator_time_step
        self.max_episode_length = max_episode_length
        self.isp = isp
        self.initial_mass = initial_mass
        self.initial_date = initial_date if initial_date is not None else pk.epoch(0)
        self.initial_spacecraft_state = None
        self.maneuver_duration = (
            integrator_time_step if maneuver_duration is None else maneuver_duration
        )
        self.max_thrust = max_thrust

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Observation space: position + velocity + mass proportion
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        # Action space: thrust (in RSW reference frame)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        (
            self.EARTH_RADIUS,
            self.GRAVITY_PARAM,
            self.SH_C,
            self.SH_S,
            _,
            _,
        ) = pk.util.load_gravity_model(gravity_model_file_path)
        self.position_normalization_factor = position_normalization_factor
        self.velocity_normalization_factor = velocity_normalization_factor
        self.trajectory = Trajectory()
        self.trajectory_log_dir = None
        if trajectory_log_dir is not None:
            self.trajectory_log_dir = Path(trajectory_log_dir)
            self.trajectory_log_dir.mkdir(parents=True, exist_ok=True)

    def _get_random_orbit(self):
        altitude_range = 400.0 * 1000, 37000.0 * 1000
        inclination_range = -180, 180
        perigee_arg_range = -180, 180
        raan_range = -180, 180
        eccentric_anomaly_range = -180, 180

        r_a, r_p = random.uniform(*altitude_range), random.uniform(*altitude_range)
        if r_a < r_p:
            r_p, r_a = r_a, r_p
        sma = (r_p + r_a + 2 * self.EARTH_RADIUS) / 2
        eccentricity = radians(1.0 - (r_p + self.EARTH_RADIUS) / sma)

        inclination = radians(random.uniform(*inclination_range))
        perigee_arg = radians(random.uniform(*perigee_arg_range))
        raan = radians(random.uniform(*raan_range))
        eccentric_anomaly = radians(random.uniform(*eccentric_anomaly_range))

        return np.array(
            [sma, eccentricity, inclination, raan, perigee_arg, eccentric_anomaly]
        )

    def _get_obs(self):
        return np.concatenate(
            [
                np.array(self.sc_position) / self.position_normalization_factor,
                np.array(self.sc_velocity) / self.velocity_normalization_factor,
                [self.sc_mass / self.initial_mass],
            ]
        )

    def _get_info(self):
        return {"spacecraft_mass": self.sc_mass, "date": self.date}

    @abstractmethod
    def _get_reward(self):
        raise NotImplementedError

    @abstractmethod
    def _is_terminated(self):
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        orbit = (
            self._get_random_orbit()
            if self.initial_orbit is None
            else self.initial_orbit
        )
        self.trajectory.clear()
        self.sc_position, self.sc_velocity = pk.par2ic(orbit, mu=self.GRAVITY_PARAM)
        self.sc_mass = self.initial_mass
        self.date = self.initial_date
        observation, info = self._get_obs(), {}
        self.prev_obs = observation
        return observation, info

    def step(self, rsw_thrust):
        norm = np.linalg.norm(rsw_thrust)
        if norm > 1.0:
            rsw_thrust /= norm
        rsw_thrust = [i.item() * self.max_thrust for i in rsw_thrust]
        thrust = rsw_to_eci(
            rsw_thrust, np.array(self.sc_position), np.array(self.sc_velocity)
        )
        self.sc_position, self.sc_velocity, self.sc_mass = pk.propagate_taylor_J2(
            self.sc_position,
            self.sc_velocity,
            self.sc_mass,
            tuple(thrust),
            self.integrator_time_step,
            self.GRAVITY_PARAM,
            self.isp * 9.8,
            self.EARTH_J2 * self.EARTH_RADIUS**2,
            -6,
            -6,
        )
        self.date = pk.epoch(
            self.date.mjd2000 + (self.integrator_time_step / (24 * 3600)), "mjd2000"
        )
        self.trajectory.log(
            self.date,
            self.prev_obs[:3] * self.position_normalization_factor,
            self.prev_obs[3:] * self.velocity_normalization_factor,
            rsw_thrust,
        )
        observation, info = self._get_obs(), self._get_info()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = False
        if self.max_episode_length is not None:
            date_offset = (self.date.mjd2000 - self.initial_date.mjd2000) * 24 * 3600.0
            truncated = date_offset > self.max_episode_length
        self.prev_obs = observation
        if (terminated or truncated) and self.trajectory_log_dir is not None:
            i = 0
            while (path := self.trajectory_log_dir / f"trajectory_{i}.json").is_file():
                i += 1
            self.trajectory.as_df().to_json(path, index=False)
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


class EarthOrbitMatchingEnv(EarthOrbitEnv):
    def __init__(
        self,
        *,
        target_orbit=None,
        weights=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_orbit = (
            target_orbit if target_orbit is not None else self._get_random_orbit()
        )
        self.weights = weights
        if weights is None:
            self.weights = np.array(5 * 0.001)

    def _get_reward(self):
        current_orbit = pk.ic2par(
            self.sc_position, self.sc_velocity, self.GRAVITY_PARAM
        )
        sma_diff = self.target_orbit[0] - current_orbit[0]
        eccentricity_diff = self.target_orbit[1] - current_orbit[1]
        inclination_diff = min(
            2 * np.pi - abs(self.target_orbit[2] - current_orbit[2]),
            abs(self.target_orbit[2] - current_orbit[2]),
        )
        raan_diff = min(
            2 * np.pi - abs(self.target_orbit[3] - current_orbit[3]),
            abs(self.target_orbit[3] - current_orbit[3]),
        )
        perigee_arg_diff = min(
            2 * np.pi - abs(self.target_orbit[4] - current_orbit[4]),
            abs(self.target_orbit[4] - current_orbit[4]),
        )

        self.element_diffs = [
            sma_diff,
            eccentricity_diff,
            inclination_diff,
            raan_diff,
            perigee_arg_diff,
        ]

        return -sum(
            weight * abs(diff) for weight, diff in zip(self.weights, self.element_diffs)
        )

    def _is_terminated(self):
        return False
