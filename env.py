class CleaningEnv:
    """
    State  = (robot position, which dirt cells are still dirty), encoded as one integer.
    Action = 0 up, 1 down, 2 left, 3 right. Moving into a wall or obstacle leaves the robot in place.
    Reward = -0.1 per step and +10 when the last cell is cleaned.
             In "dense" mode the robot also gets +1 for each cell it cleans.
    """

    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, reward_mode="dense", size=8, max_steps=200,
                 obstacles=((1, 1), (1, 2), (2, 5), (3, 3), (4, 3), (5, 6), (6, 1), (6, 2)),
                 dirt=((0, 6), (2, 2), (3, 7), (5, 4), (7, 0), (7, 6))):
        self.reward_mode = reward_mode
        self.size = size
        self.max_steps = max_steps
        self.obstacles = set(obstacles)
        self.dirt = list(dirt)
        self.n_actions = 4
        self.n_states = size * size * 2 ** len(self.dirt)

    def _state(self):
        cell = self.pos[0] * self.size + self.pos[1]
        return cell * 2 ** len(self.dirt) + self.dirty

    def reset(self):
        self.pos = (0, 0)
        self.dirty = 2 ** len(self.dirt) - 1   # one bit per dirt cell, 1 = still dirty
        self.steps = 0
        return self._state()

    def step(self, action):
        self.steps += 1
        reward = -0.1
        row = self.pos[0] + self.MOVES[action][0]
        col = self.pos[1] + self.MOVES[action][1]
        if 0 <= row < self.size and 0 <= col < self.size and (row, col) not in self.obstacles:
            self.pos = (row, col)

        if self.pos in self.dirt:
            bit = 1 << self.dirt.index(self.pos)
            if self.dirty & bit:
                self.dirty -= bit
                if self.reward_mode == "dense":
                    reward += 1.0

        cleaned = self.dirty == 0
        if cleaned:
            reward += 10.0
        done = cleaned or self.steps >= self.max_steps
        return self._state(), reward, done, cleaned

    def render(self):
        for r in range(self.size):
            line = ""
            for c in range(self.size):
                if (r, c) == self.pos:
                    line += "R "
                elif (r, c) in self.obstacles:
                    line += "# "
                elif (r, c) in self.dirt and self.dirty & (1 << self.dirt.index((r, c))):
                    line += "* "
                else:
                    line += ". "
            print(line)
        print()