# Tetris-AI
## About
This repository is intended to document the implementation of a deep RL agent to play Tetris. The following is a sample of the agent's play.

<p align="center">
    <img src="/assets/demo.gif" alt="Demo" height="30%" width="30%"/>
</p>

## Setting
The game is played on a standard 20x10 grid, with a shuffled bag of size 7 used to generate the tetrominoes (https://tetris.fandom.com/wiki/Random_Generator).

## Agent
The agent is a simple feed-forward neural network with ReLU activation.
```
agent = tf.keras.Sequential()
agent.add(tf.keras.layers.Dense(units=512, input_shape=(30,)))
agent.add(tf.keras.layers.ReLU())
agent.add(tf.keras.layers.Dense(units=256))
agent.add(tf.keras.layers.ReLU())
agent.add(tf.keras.layers.Dense(units=128))
agent.add(tf.keras.layers.ReLU())
agent.add(tf.keras.layers.Dense(units=64))
agent.add(tf.keras.layers.ReLU())
agent.add(tf.keras.layers.Dense(units=1))
```

## Features
Similar to <a href="https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/">here</a>, the features are represented by a 30D vector, comprised of the no. of complete lines (1), column-wise heights (10), column-wise holes (10), and column-wise bumpiness (9) (a one-hot encoded version of the current tetromino was initially included as part of the feature vector, but there was no substantial improvement in performance). 

## Actions
A significant complication in training an agent to play Tetris comes from the fact that an agent must learn not only *where* to place a tetromino but also *how* to get the tetromino to where it intends to place it. To partially alleviate this, we replace traditional translational/rotational actions in Tetris with *grouped* actions. In a single *grouped* action, the agent chooses a starting column and rotation for the current tetromino and then hard drops it to the current grid. Although this limits the no. of locations an agent can place a tetromino ex. the agent cannot maneuver the piece around holes/overhangs etc., this restricition is worthwhile as it allows the agent to exclusively focus on where to place the tetromino. 

## Reward
The reward received by the agent for an action is expressed as the sum of two components. The first component captures survivability. It takes a value of 1 if the game isn't over after the action, and a value of -99 if the game is over after the action. The second component captures the no. of lines cleared. Assuming that the no. of lines cleared for a given action is ```num_lines_cleared```, and the no. of columns in the grid is ```Tetris.GRID_COLS```, I initially tried setting the second component to ```num_lines_cleared * Tetris.GRID_COLS```. However, there was a substantial improvement in scoring efficiency when setting the second component to ```(num_lines_cleared ** 2) * Tetris.GRID_COLS```. This makes sense, as this approach rewards agents better for clearing more lines at one go. 

In summary,
```
reward = (1 if not game_over else -99) + (num_lines_cleared ** 2) * Tetris.GRID_COLS
```

## Performance
The following histogram shows the distribution of the no. of lines cleared by the trained agent over 30 consecutive games. The minimum and maximum no. of lines cleared in a game were 532 and 66884 respectively.

<p align="center">
    <img src="/assets/plot.png" alt="Plot" height="100%" width="100%"/>
</p>

Now we define the scoring efficiency of an agent as the average no. of lines it clears at one go. We note that the scoring efficiency is a value in the range [1, 4] (as the minimum and maximum no. of lines an agent can clear at one go are 1 and 4 respectively). Here, the agent achieved a moderate scoring efficiency of ~2 consistently over all 30 games. 

## Status
This project is still a work-in-progress.

## References
* https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
* https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
* https://github.com/uvipen/Tetris-deep-Q-learning-pytorch/

Features
Known issues -> 

What is the update? 