import tensorflow as tf

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