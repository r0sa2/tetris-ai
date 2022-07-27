from agent import agent
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
from tetris import *
from tqdm import tqdm

NUM_EPISODES: int = 4000
MAX_STEPS_PER_EPISODE: int = 10000
REPLAY_MEMORY_CAPACITY: int = 100000
REPLAY_MEMORY_MIN_SIZE: int = 4000
BATCH_SIZE: int = 2048 # 512
EPS_START: float = 1.
EPS_END: float = 0.01
EPS_DECAY: float = (EPS_START - EPS_END) / 3000
DISCOUNT: float = 0.99
RENDER_EVERY: int = 10000
TRAIN_EVERY: int = 1
SAVE_EVERY: int = 5
 
class Transition(NamedTuple):
    current_features: Features
    next_features: Features
    reward: list[Reward]
    is_game_over: list[int]

class ReplayMemory:
    def __init__(self, capacity: int=REPLAY_MEMORY_CAPACITY) -> None:
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int=BATCH_SIZE) -> list[Transition]:
        return random.sample(population=self.memory, k=batch_size)

    def clear(self) -> None:
        self.memory.clear()

    def __len__(self) -> int:
        return len(self.memory)

tetris_env: Tetris = Tetris()
agent.load_weights("./tmp/weights-episode=4550.h5")
agent.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MSE
)
replay_memory: ReplayMemory = ReplayMemory()
eps: float = EPS_END

# pbar = tqdm(range(NUM_EPISODES))
pbar = tqdm(range(4601, 6001))
episodes: list[int] = []
scores: list[int] = []
plt.ion()
plt.show()
plt.style.use("seaborn")

for episode in pbar:
    episode_steps: int = 0
    reward, current_features = tetris_env.reset()
    score: int = 0
    render: bool = episode % RENDER_EVERY == 0

    while episode_steps < MAX_STEPS_PER_EPISODE:    
        actions, rewards, features, game_overs = tetris_env.get_next_states()
        
        index = int(agent.predict(np.array(features), verbose=0).argmax())
        if random.random() <= eps:
            index = random.randint(a=0, b=len(actions) - 1)
            
        replay_memory.push(Transition(
            current_features=current_features, 
            next_features=features[index], 
            reward=rewards[index], 
            is_game_over=game_overs[index]
        ))

        if game_overs[index][0] == 0:
            episode_steps = MAX_STEPS_PER_EPISODE
            continue

        tetris_env.step(action=actions[index], render=render)
        episode_steps += 1
        current_features = features[index]
        score += rewards[index][0]

    if episode % TRAIN_EVERY == 0 and len(replay_memory) >= REPLAY_MEMORY_MIN_SIZE:
        eps = max(eps - EPS_DECAY, EPS_END)
        (
            current_features_batch,
            next_features_batch,
            reward_batch,
            is_game_over_batch
        ) = zip(*replay_memory.sample())
        agent.fit(
            np.array(current_features_batch),
            np.squeeze(reward_batch + (1 - np.array(is_game_over_batch)) * -100 + agent.predict(np.array(next_features_batch)) * is_game_over_batch, axis=1),
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=1
        )

    if episode % SAVE_EVERY == 0:
        agent.save_weights(f"./tmp/weights-episode={episode}.h5")

    pbar.set_description(f"Episode #{episode} | Score: {score} | RML: {len(replay_memory)}")
    print(score)

    # plt.cla()
    # episodes.append(episode)
    # scores.append(score)
    # plt.plot(episodes, scores)
    # plt.xlabel("Episode")
    # plt.ylabel("Score")
    # plt.draw()
    # plt.pause(0.001)