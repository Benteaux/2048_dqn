import webdriver2048 as game
import dqn
from selenium.webdriver.chrome.options import Options


chrome_options = Options()
chrome_options.add_argument('--log-level=3') # suppress the logging of any non fatal errors

# the notes are just my own ideas for future features
# note: i'm only graphing the end of episode loss, not the average loss for the episode
# note: rewards should scale exponentially, not linearly. a 64 should be worth so much more than a couple 16s.
# maybe i should take away rewards for 8s and 16s as well
# note: I may need a remove_data function to properly clear data, memory,
# model weights, epsilon decay, etc

it = 1
train = False
if train:
    game.driver = game.webdriver.Chrome(options = chrome_options)
    game.driver.get("https://play2048.co/")
    dqn.train(it, 740, reward_type = "merge", track_reward=True, random_duration = 0, avoid_wall = True, load = True, load_it = 1, load_ep = 4050)
    print(f"Iteration {it} has finished training")

dqn.load_metrics(it)
dqn.plot_metric(dqn.scores, "Scores", 1)
dqn.plot_metric(dqn.times, "Times", 2)
dqn.plot_action_frequencies(3)
dqn.plot_metric(dqn.losses, "Losses", 4)
dqn.plot_metric(dqn.rewards, "Rewards", 5)



while True:
    {}
