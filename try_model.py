from model import NeuralNetwork
from virtual_environment import Bot, Target, MinecraftVENV, get_state

nn = NeuralNetwork()
nn.load("model.json")

b = Bot()
t = Target()
env = MinecraftVENV(b, t)

# Optional: fixed positions
env.bot.set_position_override(0, 0)      # starting position
env.target.set_position_override(10, 10) # target position

max_steps = 500
for step in range(max_steps):
    state = get_state(env)  # get_state equivalent
    action = nn.decide_action(state)             # use learned model
    env.simulate_bot_action([action])           # move bot

    distance = env.euclidian_distance_to_target()
    print(f"Step {step}: distance = {distance}, angle difference = {state[2]}, action = {action}")

    if distance < 0.5:
        print(f"Target reached in {step+1} steps!")
        break
else:
    print("Target not reached within max steps.")
