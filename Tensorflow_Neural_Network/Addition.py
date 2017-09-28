import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

X=tf.placeholder(tf.float32,name="X")
Y=tf.placeholder(tf.float32, name="Y")

addition=tf.add(X,Y,name="Addition")

with tf.Session() as session:
    result=session.run(addition,feed_dict={X:[1,4,5,6,20], Y: [4,5,7,1,10]})
    print(result)
