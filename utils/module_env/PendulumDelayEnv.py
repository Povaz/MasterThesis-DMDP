import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path
from utils.delays import NormedPositiveCompoundBernoulliProcess, ConstantDelay


class PendulumDelayEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, delay=0, stochastic_delays=False, p_delay=0.70, max_delay=50):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        # Action Space definition
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )

        # State Space definition
        # Note: Here State Space is not the actual State Space but rather the Observation Space
        # without Delay integrated. It's called State Space for Compatibility purposes with
        # Recurrent Network Layer implementation
        self.high_obs = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.state_space = spaces.Box(
            low=-self.high_obs,
            high=self.high_obs,
            dtype=np.float32
        )

        # Delay Variables initialization
        self._hidden_obs = None
        self._reward_stock = None
        self.extended_obs = None
        if stochastic_delays:
            self.delay = NormedPositiveCompoundBernoulliProcess(p_delay, delay, max_value=max_delay)
        else: 
            self.delay = ConstantDelay(delay)

        # observation
        high_ext_obs = np.concatenate((self.high_obs,
                                       np.full(self.delay.max, self.max_torque))).astype(np.float32)
        low_ext_obs = np.concatenate((-self.high_obs,
                                      np.full(self.delay.max, -self.max_torque))).astype(np.float32)
        self.observation_space = spaces.Box(
            low=low_ext_obs,
            high=high_ext_obs,
            dtype=np.float32
        )

        self.reward_range = np.array([-17., 0.], dtype=np.float32)

        self.seed()

        # Other Variables useful during execution of the Environment
        self.np_random = None
        self.state = None
        self.last_u = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(action, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        
        # -------- Handling the delay --------
        # Sample new2 delay
        _, n_obs = self.delay.sample()
        # Get current state
        self._hidden_obs.append(self._get_obs())

        # Update extended state, rewards and hidden variables
        self.extended_obs.append(action)
        hidden_output = None
        if n_obs > 0:
            self.extended_obs[0] = self._hidden_obs[n_obs]
            del self.extended_obs[1:(1+n_obs)]
            hidden_output = self._hidden_obs[1:(1+n_obs)]
            del self._hidden_obs[:n_obs]
        self._reward_stock = np.append(self._reward_stock, -costs)
        reward_output = self._reward_stock[:n_obs]
        self._reward_stock = np.delete(self._reward_stock, range(n_obs))

        # Shaping the output
        try:
            output = np.concatenate(self.extended_obs)
        except:
            output = self.extended_obs
        return output, reward_output, False, (n_obs, hidden_output)

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = np.random.uniform(low=-high, high=high)
        self.last_u = None
        
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

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human', debug=False):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
