import gym


class RoomEnv(gym.Env):
    # DO NOT CHANGE. ENV IS ALREADY COMPLETED.

    # Row major map (0-based indices).
    MAP_ = [
        "S    OO ",
        "     OO ",
        "O       ",
        "        ",
        "    O   ",
        "  OOO   ",
        "        ",
        "  O    G",
    ]

    def __init__(self):
        self.max_timesteps = 500
        self.reset()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(8*8)

    def step(self, action):
        orig_pos = self.robot_pos[:]

        self.ts += 1
        # up.
        if action == 0:
            self.robot_pos[0] -= 1
        # right
        elif action == 1:
            self.robot_pos[1] += 1
        # down
        elif action == 2:
            self.robot_pos[0] += 1
        # left
        elif action == 3:
            self.robot_pos[1] -= 1
        else:
            raise ValueError("bad action!")

        # Reward/done function.
        reward = -0.1
        done = False

        # Check room boundaries and obstacles.
        if self.robot_pos[0] < 0 or self.robot_pos[0] >= 8 or \
                self.robot_pos[1] < 0 or self.robot_pos[1] >= 8 or \
                self.MAP_[self.robot_pos[0]][self.robot_pos[1]] == "O":
            self.robot_pos = orig_pos
            reward = -1.0

        # Check, whether we have reached the goal.
        if self.MAP_[self.robot_pos[0]][self.robot_pos[1]] == "G":
            reward = 10.0
            done = True

        # Determine, whether episode is over.
        if done is False:
            done = self.ts >= self.max_timesteps

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.ts = 0
        self.robot_pos = [0, 0]
        return self._get_obs()

    def _get_obs(self):
        int_pos = self.robot_pos[0] * 8 + self.robot_pos[1]
        return int_pos


if __name__ == "__main__":
    # Test the env.
    env = RoomEnv()
    env.reset()
    done = False
    episode_reward = 0.0
    while not done:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        episode_reward += reward

    print(f"Reward={episode_reward} after 1 episode.")
