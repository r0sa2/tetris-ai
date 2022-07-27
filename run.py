from agent import agent
from math import sqrt
from tetris import *

tetris_env: Tetris = Tetris()
agent.load_weights("./tmp/weights-episode=4875.h5")

_, current_features = tetris_env.reset()
lines_cleared: int = 0
score: int = 0
score_threshold: int = 2500
while True:
    actions, rewards, features, game_overs = tetris_env.get_next_states()
    index = agent.predict(np.array(features), verbose=0).argmax()
    tetris_env.step(action=actions[index], render=False)
    lines_cleared += int(sqrt((rewards[index][0] - 1) // Tetris.GRID_COLS))
    score += rewards[index][0]

    if score > score_threshold:
        print(f"(lines_cleared, score): ({lines_cleared}, {score})")
        score_threshold += 500
    if game_overs[index][0] == 1:
        current_features = features[index]
    else:
        _, current_features = tetris_env.reset()
        lines_cleared = 0
        score = 0
        score_threshold = 0

# (9268, 163006)
# (9393, 164001)
# (18159, 316011)
# (1395, 24531)
# (6371, 111501)
# (3759, 65565)
# (3833, 67006)
# (9893, 173010)
# (9394, 163022)
# (1606, 28001)

# (26680, 598008)