# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding
from utils.delays import NormedPositiveCompoundBernoulliProcess, ConstantDelay


class MountainCarDelayEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, delay=0, goal_velocity=0, stochastic_delays=False, p_delay=0.70, max_delay=50):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        # Delay Variables initialization
        self._hidden_obs = None
        self._reward_stock = None
        self.extended_obs = None
        if stochastic_delays:
            self.delay = NormedPositiveCompoundBernoulliProcess(p_delay, delay, max_value=max_delay)
        else: 
            self.delay = ConstantDelay(delay)

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        self.state_space = spaces.Box(self.low_state, self.high_state, dtype=np.float32)

        # Delayed observation space
        high_state = np.concatenate((self.high_state, np.full(self.delay.max, self.max_action))).astype(np.float32)
        low_state = np.concatenate((self.low_state, np.full(self.delay.max, self.min_action))).astype(np.float32)

        self.observation_space = spaces.Box(
            low=low_state,
            high=high_state,
            dtype=np.float32
        )

        # Render Variables
        self.render_action = None
        self.render_value = None

        # Reward range
        self.reward_range = np.array([-1.,0.], dtype=np.float32)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Save the Action for the render
        self.render_action = action[0]

        # PyTorch Bug Print
        if np.isnan(action[0]):
            print('NAN Action')

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed: velocity = self.max_speed
        if velocity < -self.max_speed: velocity = -self.max_speed
        position += velocity
        if position > self.max_position: position = self.max_position
        if position < self.min_position: position = self.min_position
        if position == self.min_position and velocity < 0: velocity = 0

        # Convert a possible numpy bool to a Python bool.
        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        self.state = np.array([position, velocity])

        # Sample new2 delay
        _, n_obs = self.delay.sample()

        # Get current state
        self._hidden_obs.append(self.state)

        # Update extended state, rewards and hidden variables
        self.extended_obs.append(action)
        hidden_output = None
        if n_obs > 0:
            self.extended_obs[0] = self._hidden_obs[n_obs]
            del self.extended_obs[1:(1 + n_obs)]
            hidden_output = self._hidden_obs[1:(1 + n_obs)]
            del self._hidden_obs[:n_obs]

        # reward
        reward = 0
        reward -= math.pow(action[0], 2) * 0.1
        self._reward_stock = np.append(self._reward_stock, reward)
        if done:
            self._reward_stock[-1] += 100.0
            reward_output = np.sum(self._reward_stock)
            # reward_output = self._reward_stock # -> in this case, the sum is to be done in the algorithm
        else:
            reward_output = self._reward_stock[:n_obs]
        self._reward_stock = np.delete(self._reward_stock, range(n_obs))

        # Shaping the output
        try:
            output = np.concatenate(self.extended_obs)
        except:
            output = self.extended_obs
        return output, reward_output, done, (n_obs, hidden_output)

    def reset(self):
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])

        self.delay.reset()
        # Let the Agent act to fill the Extended Observation caused by the presence of Delay
        self._hidden_obs = [0 for _ in range(self.delay.current)]
        self.extended_obs = [0 for _ in range(self.delay.current+1)]
        self._reward_stock = np.array([0 for _ in range(self.delay.current)])
        self._hidden_obs.append(self.state)
        self.extended_obs[0] = self._hidden_obs[-1]
        obs_before_start = self.delay.current
        while obs_before_start > 0:
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

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human', debug=False):
        screen_width = 600
        screen_height = 400

        if debug:
            screen_width = 1200
            screen_height = 800

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth = 40
        carheight = 20

        from gym.envs.classic_control import rendering
        if self.viewer is None:

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

            if debug:
                # Policy Action
                action_start = 0.0
                action_height = 0.04
                action_end = action_start + self.render_action
                action_x = np.array([action_start * scale, action_end * scale])
                action_y = np.array([action_height * scale, action_height * scale])
                action_line = rendering.make_polyline(list(zip(action_x, action_y)))
                action_line.set_color(1.0, 0.0, 0.0)
                action_line.set_linewidth(3)
                action_line.add_attr(self.cartrans)
                self.viewer.add_geom(action_line)

                # Value State
                state_start = 0.0 
                state_end = state_start + self.render_value
                state_x = np.array([0.0 * scale, 0.0 * scale])
                state_y = np.array([state_start * scale, state_end * scale])
                state_line = rendering.make_polyline(list(zip(state_x, state_y)))
                state_line.set_color(0.0, 1.0, 0.0)
                state_line.set_linewidth(2)
                state_line.add_attr(self.cartrans)
                self.viewer.add_geom(state_line)

                # State[1]: Speed
                speed_start = 0.0
                speed_height = 0.05
                speed_end = speed_start + self.state[1]
                speed_x = np.array([speed_start * scale, speed_end * scale])
                speed_y = np.array([speed_height * scale, speed_height * scale])
                speed_line = rendering.make_polyline(list(zip(speed_x, speed_y)))
                speed_line.set_color(0.0, 0.0, 1.0)
                speed_line.set_linewidth(3)
                speed_line.add_attr(self.cartrans)
                self.viewer.add_geom(speed_line)

        if debug:
            # Policy Action
            action_start = 0.0
            action_height = 0.04
            action_end = action_start + self.render_action
            action_x = np.array([action_start * scale, action_end * scale])
            action_y = np.array([action_height * scale, action_height * scale])
            action_line = rendering.make_polyline(list(zip(action_x, action_y)))
            action_line.set_color(1.0, 0.0, 0.0)
            action_line.set_linewidth(3)
            action_line.add_attr(self.cartrans)
            self.viewer.geoms[-3] = action_line

            # Value State
            state_start = 0.0
            state_height = 0.0
            state_scale = 1e-4
            state_end = state_start + self.render_value * state_scale
            state_x = np.array([state_height * scale, state_height * scale])
            state_y = np.array([state_start * scale, state_end * scale])
            state_line = rendering.make_polyline(list(zip(state_x, state_y)))
            state_line.set_color(0.0, 1.0, 0.0)
            state_line.set_linewidth(2)
            state_line.add_attr(self.cartrans)
            self.viewer.geoms[-2] = state_line

            # State[1]: Speed
            speed_start = 0.0
            speed_height = 0.05
            speed_scale = 5.0
            speed_end = speed_start + self.state[1] * speed_scale
            speed_x = np.array([speed_start * scale, speed_end * scale])
            speed_y = np.array([speed_height * scale, speed_height * scale])
            speed_line = rendering.make_polyline(list(zip(speed_x, speed_y)))
            speed_line.set_color(0.0, 0.0, 1.0)
            speed_line.set_linewidth(3)
            speed_line.add_attr(self.cartrans)
            self.viewer.geoms[-1] = speed_line

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render_learned(self, mode='human'):
        from gym.envs.classic_control import rendering

        # Window Parameters
        screen_width = 1200
        screen_height = 800
        world_width = self.max_position - self.min_position
        scale = screen_width/world_width

        # Create the Viewer Object for the first time along with Fixed Objects in the Drawing
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # MountainCar Track
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            # MountainCar Flag (Top of the Hill)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        # MountainCar Transformation
        cartrans = rendering.Transform()

        # MountainCar Position
        radius = 0.01
        pos_circle = rendering.make_circle(radius*scale, filled=True)
        pos_circle.set_color(0.0, 0.0, 1.0)
        pos_circle.add_attr(cartrans)
        self.viewer.add_geom(pos_circle)

        # Policy Action
        action_start = 0.0
        action_end = action_start + self.render_action*0.5
        action_x = np.array([action_start*scale, action_end*scale])
        action_y = np.array([0.0*scale, 0.0*scale])
        action_line = rendering.make_polyline(list(zip(action_x, action_y)))
        action_line.set_color(1.0, 0.0, 0.0)
        action_line.set_linewidth(2)
        action_line.add_attr(cartrans)
        self.viewer.add_geom(action_line)

        # Value State
        state_start = 0.0
        state_end = state_start + self.render_value*1e-6
        state_x = np.array([0.0*scale, 0.0*scale])
        state_y = np.array([state_start*scale, state_end*scale])
        state_line = rendering.make_polyline(list(zip(state_x, state_y)))
        state_line.set_color(0.0, 1.0, 0.0)
        state_line.set_linewidth(2)
        state_line.add_attr(cartrans)
        self.viewer.add_geom(state_line)

        # Calculate Position from the state
        pos = self.state[0]
        cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def continue_render(self, mode='human'):
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')