import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
value=tf.constant("Hello-World!!!!")
session=tf.Session()
print(session.run(value))
