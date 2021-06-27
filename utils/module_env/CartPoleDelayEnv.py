"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym.utils import seeding
from gym import spaces, logger
import numpy as np
from utils.delays import NormedPositiveCompoundBernoulliProcess, ConstantDelay


class CartPoleDelayEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, delay=0, stochastic_delays=False, p_delay=0.7, max_delay=50):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.max_force = 20
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Delay Variables initialization
        self._hidden_obs = None
        self._reward_stock = None
        self.extended_obs = None
        if stochastic_delays:
            self.delay = NormedPositiveCompoundBernoulliProcess(p_delay, delay, max_value=max_delay)
        else: 
            self.delay = ConstantDelay(delay, max_value=max_delay)

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,))
        self.state_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.concatenate((high, np.full(self.delay.max, self.max_force))).astype(np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.state = None

        # Seeding
        self.np_random = None

        # Render Variable
        self.viewer = None

        # Reward range
        self.reward_range = np.array([1.,1.], dtype=np.float32)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def state_eq(self, st, u):
        x, x_dot, theta, theta_dot = st
        force = u[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return np.array([x, x_dot, theta, theta_dot])

    def step(self, action):
        # Error Message for Invalid Actions
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # Given the Action, compute the next state of the Model
        self.state = self.state_eq(self.state, action)
        x, x_dot, theta, theta_dot = self.state

        # Define Done Signal
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            # If the Episode is not finished, set action reward to 1
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            # Warn the User about calling Step after the Episode is finished, without resetting it first
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        # -------- Handling the delay --------
        # Sample new2 delay
        _, n_obs = self.delay.sample()
        # Get current state
        self._hidden_obs.append(self.state)

        # Update extended state, rewards and hidden variables
        self.extended_obs.append(action)
        hidden_output = None
        if n_obs>0:
            self.extended_obs[0] = self._hidden_obs[n_obs]
            del self.extended_obs[1:(1+n_obs)]
            hidden_output = self._hidden_obs[1:(1+n_obs)]
            del self._hidden_obs[:n_obs]
        self._reward_stock = np.append(self._reward_stock, reward)
        reward_output = self._reward_stock[:n_obs]
        self._reward_stock = np.delete(self._reward_stock, range(n_obs))

        # Shaping the output
        try:
            output = np.concatenate(self.extended_obs)
        except:
            output = self.extended_obs
        return output, reward_output, False, (n_obs, hidden_output)

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None

        self.delay.reset()
        # Let the Agent act to fill the Extended Observation caused by the presence of Delay
        self._hidden_obs = [0 for _ in range(self.delay.current)]
        self.extended_obs = [0 for _ in range(self.delay.current+1)]
        self._reward_stock = np.array([0 for _ in range(self.delay.current)])
        self._hidden_obs.append(self._get_obs())
        self.extended_obs[0] = self._hidden_obs[-1]
        obs_before_start = self.delay.current
        while obs_before_start>0:
            action = self.action_space.sample()
            _, _, done, info = self.step(action)
            if done:
                logger.warn("The environment failed before delay timesteps.")
                break
            obs_before_start -= info[0]

        try:
            return np.concatenate(self.extended_obs)
        except:
            return np.array(self.extended_obs)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None