import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from agent import Agent

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2', render_mode="human")
    env = gym.make("Humanoid-v4", render_mode="human")
    agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300, 
                    n_actions=env.action_space.shape[0])
    n_games = 10000
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        #agent.noise.reset()
        while not done:
            action = agent.policy(observation)
            observation_, reward, done, info, _ = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
