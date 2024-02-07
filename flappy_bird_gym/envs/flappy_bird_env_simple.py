#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Implementation of a Flappy Bird OpenAI Gym environment that yields simple
numerical information about the game's state as observations.
"""

from typing import Dict, Tuple, Optional, Union

import gym
import numpy as np
import pygame

from flappy_bird_gym.envs.game_logic import FlappyBirdLogic
from flappy_bird_gym.envs.game_logic import PIPE_WIDTH, PIPE_HEIGHT
from flappy_bird_gym.envs.game_logic import PLAYER_WIDTH, PLAYER_HEIGHT
from flappy_bird_gym.envs.renderer import FlappyBirdRenderer


class FlappyBirdEnvSimple(gym.Env):
    """ Flappy Bird Gym environment that yields simple observations.

    The observations yielded by this environment are simple numerical
    information about the game's state. Specifically, the observations are:

        * Horizontal distance to the next pipe;
        * Difference between the player's y position and the next hole's y
          position.

    The reward received by the agent in each step is equal to the score obtained
    by the agent in that step. A score point is obtained every time the bird
    passes a pipe.

    Args:
        screen_size (Tuple[int, int]): The screen's width and height.
        normalize_obs (bool): If `True`, the observations will be normalized
            before being returned.
        pipe_gap (int): Space between a lower and an upper pipe.
        bird_color (str): Color of the flappy bird. The currently available
            colors are "yellow", "blue" and "red".
        pipe_color (str): Color of the pipes. The currently available colors are
            "green" and "red".
        background (Optional[str]): Type of background image. The currently
            available types are "day" and "night". If `None`, no background will
            be drawn.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 screen_size: Tuple[int, int] = (288, 512),
                 normalize_obs: bool = True,
                 pipe_gap: int = 100,
                 bird_color: str = "yellow",
                 pipe_color: str = "green",
                 background: Optional[str] = "day") -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(6,),
                                                dtype=np.float32)
        self._screen_size = screen_size
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap

        self._game = None
        self._renderer = None

        self._bird_color = bird_color
        self._pipe_color = pipe_color
        self._bg_type = background

        self._y_tmin2 = 0
        self._y_tmin1 = 0

    def _get_pos(self):
        player_y = self._game.player_y
        h_dists = np.array([np.nan, np.nan, np.nan], dtype=np.float32)   # Assumes there's only three pipes ever
        v_dists = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        for i, (up_pipe, low_pipe) in enumerate(zip(self._game.upper_pipes, self._game.lower_pipes)):
            upper_pipe_y = up_pipe["y"] + PIPE_HEIGHT
            lower_pipe_y = low_pipe["y"]
            h_dists[i] = (low_pipe["x"] + PIPE_WIDTH / 2 - (self._game.player_x - PLAYER_WIDTH / 2)) + 3  # extra distance to compensate for buggy hit-box
            v_dists[i] = (upper_pipe_y + lower_pipe_y) / 2 - (player_y + PLAYER_HEIGHT/2)

        idxs = np.argsort(h_dists)
        h_dists, v_dists = h_dists[idxs], v_dists[idxs]
        h_dists, v_dists = h_dists[h_dists >= 0], v_dists[h_dists >= 0] # remove nans and past pipes

        if self._normalize_obs:
            h_dists /= self._screen_size[0]
            v_dists /= self._screen_size[1]

        if len(h_dists) == 2:
            return np.concatenate([h_dists, v_dists])
        else:
            return np.array([h_dists[0], -1, v_dists[0], -1])

    def _get_obs(self):
        delta_min1, delta_now = self._y_tmin1 - self._y_tmin2, self._y_tmin1 - self._game.player_y
        if self._normalize_obs:
            delta_min1 /= self._screen_size[1]
            delta_now /= self._screen_size[1]
        return np.append(
            self._get_pos(),
            [delta_min1, delta_now]
        )

    def step(self,
             action: Union[FlappyBirdLogic.Actions, int],
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """ Given an action, updates the game state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The action taken by
                the agent. Zero (0) means "do nothing" and one (1) means "flap".

        Returns:
            A tuple containing, respectively:

                * an observation (horizontal distance to the next pipe;
                  difference between the player's y position and the next hole's
                  y position);
                * a reward (always 1);
                * a status report (`True` if the game is over and `False`
                  otherwise);
                * an info dictionary.
        """
        self._y_tmin2 = self._y_tmin1
        self._y_tmin1 = self._game.player_y
        alive = self._game.update_state(action)
        obs = self._get_obs()

        if alive:
            if action == 0:
                reward = 1.
            else:
                reward = 0.5
        else:
            reward = 0.

        done = not alive
        info = {"score": self._game.score}

        return obs, reward, done, info

    def reset(self):
        """ Resets the environment (starts a new game). """
        self._y_tmin2 = 0
        self._y_tmin1 = 0
        self._game = FlappyBirdLogic(screen_size=self._screen_size,
                                     pipe_gap_size=self._pipe_gap)
        if self._renderer is not None:
            self._renderer.game = self._game

        self.step(0)
        return self.step(0)[0]

    def render(self, mode='human') -> None:
        """ Renders the next frame. """
        if self._renderer is None:
            self._renderer = FlappyBirdRenderer(screen_size=self._screen_size,
                                                bird_color=self._bird_color,
                                                pipe_color=self._pipe_color,
                                                background=self._bg_type)
            self._renderer.game = self._game

        self._renderer.draw_surface(show_score=True)
        if mode == "rgb_array":
            return pygame.surfarray.array3d(self._renderer.surface)
        else:
            if self._renderer.display is None:
                self._renderer.make_display()
            self._renderer.update_display()

    def close(self):
        """ Closes the environment. """
        if self._renderer is not None:
            pygame.display.quit()
            self._renderer = None
        super().close()
