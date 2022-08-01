from agent import agent
from math import sqrt
from tetris import *

agent.load_weights("model.h5")
tetris_env: Tetris = Tetris()
_, current_features = tetris_env.reset()
lines_cleared: int = 0
score: int = 0
score_print_threshold: int = 2500

while True:
    actions, rewards, features, game_overs = tetris_env.get_next_states()
    index = agent.predict(np.array(features), verbose=0).argmax()
    tetris_env.step(action=actions[index], render=True)
    lines_cleared += int(sqrt((rewards[index][0] - 1) // Tetris.GRID_COLS))
    score += rewards[index][0]

    if score > score_print_threshold:
        print(f"(lines_cleared, score): ({lines_cleared}, {score})")
        score_print_threshold += 2500
    if game_overs[index][0] == 1:
        current_features = features[index]
    else:
        print(f"(lines_cleared, score): ({lines_cleared}, {score})")
        _, current_features = tetris_env.reset()
        lines_cleared = 0
        score = 0
        score_print_threshold = 0
