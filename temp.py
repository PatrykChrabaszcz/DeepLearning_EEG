import tensorflow as tf
import numpy as np

summary_writer = tf.summary.FileWriter('logs/%')

for i in range(10):
    summary = tf.Summary(value=[tf.Summary.Value(tag='RandomVariable', simple_value=np.random.uniform(0, 10))])
    summary_writer.add_summary(summary, i)

summary_writer.flush()

