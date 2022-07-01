from collections import deque
import tensorflow as tf
from tetris import *
from tqdm import tqdm
from typing import Optional

REPLAY_MEMORY_CAPACITY: int = 20000
BATCH_SIZE: int = 512
NUM_EPISODES: int = 2000
MAX_STEPS_PER_EPISODE: int = 1000
RENDER_EVERY: int = 10
TRAIN_EVERY: int = 10
EPS_START: float = 0.99
EPS_END: float = 0.0
EPS_DECAY: float = (EPS_START - EPS_END) / 1000
DISCOUNT: float = 0.99
GAME_OVER_PENALTY: Reward = -100

class Transition(NamedTuple):
    current_features: Features
    next_features: Optional[Features]
    reward: Reward
    is_game_over: bool

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

agent = tf.keras.Sequential()
agent.add(tf.keras.layers.Dense(units=512, activation="relu", input_shape=(Tetris.GRID_COLS * 2 + Tetromino.NUM_TETROMINOS,)))
agent.add(tf.keras.layers.Dense(units=256, activation="relu"))
agent.add(tf.keras.layers.Dense(units=128, activation="relu"))
agent.add(tf.keras.layers.Dense(units=1))

agent.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MSE
)

tetris_env = Tetris()
replay_memory = ReplayMemory()
eps: float = EPS_START
pbar = tqdm(range(NUM_EPISODES))

for episode in pbar:
    episode_steps: int = 0
    reward, current_features = tetris_env.reset()
    score: Reward = 0
    render: bool = episode % RENDER_EVERY == 0

    while episode_steps < MAX_STEPS_PER_EPISODE:
        actions, rewards, features = tetris_env.get_next_states()

        if len(actions) == 0:
            replay_memory.push(Transition(current_features=current_features, next_features=None, reward=GAME_OVER_PENALTY, is_game_over=True))
            episode_steps = MAX_STEPS_PER_EPISODE
        else:
            if random.random() <= eps:
                index = random.randint(a=0, b=len(actions) - 1)
            else:
                index = (rewards + DISCOUNT * agent.predict(np.array(features), verbose=0)).argmax()
                replay_memory.push(Transition(current_features=current_features, next_features=features[index], reward=rewards[index][0], is_game_over=False))

            tetris_env.step(action=actions[index], render=render)
            score += rewards[index][0]
            current_features = features[index]
            episode_steps += 1

    if episode % TRAIN_EVERY == 0 and len(replay_memory) > BATCH_SIZE:
        batch: list[Transition] = replay_memory.sample()
        x = np.array([transition.current_features for transition in batch])
        y = np.array([transition.reward + (agent.predict(np.expand_dims(a=transition.next_features, axis=0), verbose=0)[0][0] if not transition.is_game_over else 0) for transition in batch])
        agent.fit(x, y, batch_size=BATCH_SIZE, epochs=1, verbose=0)

    eps = max(eps - EPS_DECAY, EPS_END)

    pbar.set_description(f"Episode #{episode} | Score: {score} | RML: {len(replay_memory)}")
