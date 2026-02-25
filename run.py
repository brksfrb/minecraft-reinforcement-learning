from virtual_environment import Bot, Target, MinecraftVENV, get_state
from model import NeuralNetwork

b = Bot()
t = Target()
env = MinecraftVENV(b, t)
env.randomize_target_position()
env.randomize_bot_position()

nn = NeuralNetwork()
episodes = 10000
max_steps = 50

for ep in range(episodes):
    if ep%1000 == 0:
        print("Episode: ", ep)
    env.randomize_bot_position()
    env.randomize_target_position()
    old_distance = env.euclidian_distance_to_target()
    old_angle = abs(get_state(env)[2])  # initial angle_to_target

    epsilon = max(0.01, 0.5 * (1 - ep/episodes))  # Epsilon decay
    for step in range(max_steps):
        state = get_state(env)
        action = nn.decide_action(state, epsilon)  # epsilon-greedy
        env.simulate_bot_action([action])

        new_distance = env.euclidian_distance_to_target()
        new_angle = abs(get_state(env)[2])

        # Calculate distance reward (positive if bot gets closer)
        distance_reward = max(0, old_distance - new_distance)

        # Calculate angle reward (positive if bot turns toward the target)
        old_angle = abs(old_angle) % 360
        new_angle = abs(new_angle) % 360
        # Ensure shortest angular difference
        angle_diff = old_angle - new_angle
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        angle_reward = max(0, angle_diff)

        # Optional: bonus for reaching target
        target_reward = 10 if new_distance < 0.5 else 0

        # Total reward
        reward = distance_reward + 0.5 * angle_reward + target_reward

        old_distance = new_distance
        old_angle = new_angle

        nn.train(state, reward)

        if new_distance < 0.5:
            break


nn.save("model.json")
