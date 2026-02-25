import math
import random


class Bot:
    def __init__(self):
        self._position: list[float] = [0, 0]
        self._rotation = 0
    def set_position(self, x, y):
        if abs(self.x - x) > 1 or abs(self.y - y) > 1:
            print("Too far.")
            return
        self._position = [x, y]
    def move_forward(self):
        self._position[0] += math.sin(math.radians(self.rotation))
        self._position[1] += math.cos(math.radians(self.rotation))
    def rotate_bot(self, amount: int):
        self._rotation += amount
        self._rotation = (self.rotation + 180) % 360 - 180

    def set_position_override(self, x, y):
        self._position = [x, y]

    @property
    def x(self):
        return self._position[0]

    @property
    def y(self):
        return self._position[1]

    @property
    def rotation(self):
        return self._rotation

class Target:
    def __init__(self):
        self._position: list[float] = [0, 0]

    def set_position(self, x, y):
        if abs(self.x - x) > 1 or abs(self.y - y) > 1:
            print("Too far.")
            return
        self._position = (x, y)

    def set_position_override(self, x, y):
        self._position = (x, y)

    @property
    def x(self):
        return self._position[0]

    @property
    def y(self):
        return self._position[1]

class MinecraftVENV:
    def __init__(self, bot, target):
        self.bot = bot
        self.target = target
    def randomize_bot_position(self):
        self.bot.set_position_override(random.randrange(1, 50), random.randrange(1, 50))
    def randomize_target_position(self):
        self.target.set_position_override(random.randrange(1, 50), random.randrange(1, 50))
    def euclidian_distance_to_target(self):
        dx = self.bot.x - self.target.x
        dy = self.bot.y - self.target.y
        euclidian_distance = math.sqrt(dx**2 + dy**2)
        return euclidian_distance
    def manhattan_distance_to_target(self):
        dx = self.bot.x - self.target.x
        dy = self.bot.y - self.target.y
        manhattan_distance = abs(dx) + abs(dy)
        return manhattan_distance
    def distances(self):
        dx = self.bot.x - self.target.x
        dy = self.bot.y - self.target.y
        return dx, dy
    def target_origin_angle(self):
        dx = self.bot.x - self.target.x
        dy = self.bot.y - self.target.y
        if dy == 0:
            if dx >= 0: return 90
            else: return -90
        if dx == 0:
            if dy >= 0:
                if dy >= 0: return 0
                else: return 180
        if dy > 0:
            angle = math.atan(abs(dx)/dy)
            if dx<0: angle = -angle
            return angle
        if dy <0:
            angle = 180 - math.atan(abs(dx)/abs(dy))
            if dx<0: angle = -angle
            return angle

    def angle_difference(self):
        ab = self.bot.rotation
        at = self.target_origin_angle()
        dif = at - ab
        if abs(dif) > 180:
            dif = 180 - (abs(dif) % 180)
        if ab < 0:
            dif = -dif
        return dif
    def simulate_bot_action(self, actions: list):
        for action in actions:
            if action == "move_forward":
                self.bot.move_forward()
            elif action == "rotate_15_right":
                self.bot.rotate_bot(15)
            elif action == "rotate_15_left":
                self.bot.rotate_bot(-15)

def get_state(env, max_distance=50):
    dx, dy = env.distances()
    dx = max(-max_distance, min(max_distance, dx))
    dy = max(-max_distance, min(max_distance, dy))

    dx_norm = dx / max_distance
    dy_norm = dy / max_distance

    # Compute angle-to-target relative to bot's current rotation
    angle_to_target = math.degrees(math.atan2(dy, dx)) - env.bot.rotation
    # Wrap angle to [-180, 180]
    angle_to_target = (angle_to_target + 180) % 360 - 180
    angle_norm = angle_to_target / 180  # normalize to [-1, 1]

    return [dx_norm, dy_norm, angle_norm]

# Example Usage
"""
b = Bot()
t = Target()
demo_env = MinecraftVENV(b, t)
demo_env.randomize_bot_position()
demo_env.randomize_target_position()
while True:
    print(b.x, b.y, b.rotation, "/", t.x, t.y)
    act = input("Enter action: ")
    act_s = input("Enter secondary action(Optional): ")
    demo_env.simulate_bot_action([act, act_s])
"""