import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

nodes_layerone=50
nodes_layertwo=100
nodes_layerthree=50
epochs=100
learning_rate=0.001
number_of_inputs=9
number_of_outputs=1

training_dataset=pd.read_csv("Datasets\\training\sales_data_training.csv")
X_training_inputs=training_dataset.drop('total_earnings',axis=1).values
Y_training_outputs=training_dataset[['total_earnings']].values

testing_dataset=pd.read_csv("Datasets\\testing\sales_data_test.csv")
X_test_inputs=training_dataset.drop('total_earnings',axis=1).values
Y_test_outputs=training_dataset[['total_earnings']].values




#Normalization or scaling
X_scale=MinMaxScaler(feature_range=(0,1))
Y_scale=MinMaxScaler(feature_range=(0,1))

X_scaled_training=X_scale.fit_transform(X_training_inputs)
Y_scaled_training=Y_scale.fit_transform(Y_training_outputs)

X_scaled_test=X_scale.fit_transform(X_test_inputs)
Y_scaled_test=Y_scale.fit_transform(Y_test_outputs)

print(X_scaled_test.shape)
print(Y_scaled_test.shape)

print("The constants multiplied to scale y is {:.10f} and added is {:.02f}".format(Y_scale.scale_[0],Y_scale.min_[0]))


#Defining layers of neural network

#Input Layer
with tf.variable_scope('input'):
    X=tf.placeholder(tf.float32,shape=(None,number_of_inputs))

#Layer 1
with tf.variable_scope('layer_1'):
    weights=tf.get_variable('weights1',shape=[number_of_inputs,nodes_layerone],initializer=tf.contrib.layers.xavier_initializer())
    biases=tf.get_variable('biases1', initializer=tf.zeros_initializer(shape=[nodes_layerone]))
    layer_1_output=tf.nn.relu(tf.matmul(X,weights)+biases)

#Layer 2
with tf.variable_scope('layer_2'):
    weights=tf.get_variable('weights2',shape=[nodes_layerone,nodes_layertwo],initializer=tf.contrib.layers.xavier_initializer())
    biases=tf.get_variable('biases2', initializer=tf.zeros_initializer(shape=[nodes_layertwo]))
    layer_2_output=tf.nn.relu(tf.matmul(layer_1_output,weights)+biases)

#Layer 3
with tf.variable_scope('layer_3'):
    weights=tf.get_variable('weights3',shape=[nodes_layertwo,nodes_layerthree],initializer=tf.contrib.layers.xavier_initializer())
    biases=tf.get_variable('biases3', initializer=tf.zeros_initializer(shape=[nodes_layerthree]))
    layer_3_output=tf.nn.relu(tf.matmul(layer_2_output,weights)+biases)

#Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable('weights4', shape=[nodes_layerthree,number_of_outputs],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases4', initializer=tf.zeros_initializer( shape=[number_of_outputs]))
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

#Loss/ Cost function
with tf.variable_scope('cost'):
    Y=tf.placeholder(tf.float32,shape=(None,number_of_outputs))
    cost=tf.reduce_mean(tf.squared_difference(prediction,Y))

#Optimize cost
with tf.variable_scope('train'):
    optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Write logs

with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost',cost)
    summary=tf.summary.merge_all()
    saver=tf.train.Saver()
#Session
with tf.Session() as session:

    # session.run(tf.global_variables_initializer())
    #
    # training_writer=tf.summary.FileWriter(".\logs\\training",session.graph)
    # test_writer = tf.summary.FileWriter(".\logs\\test", session.graph)
    # for epoch in range(epochs):
    #
    #     session.run(optimizer, feed_dict={X: X_scaled_training, Y:Y_scaled_training})
    #     # print("Training iteration: {}".format(epoch))
    #     if epoch % 5==0:
    #         training_cost,training_summary=session.run([cost,summary], feed_dict={X: X_scaled_training, Y:Y_scaled_training})
    #         testing_cost,test_summary=session.run([cost,summary], feed_dict={X: X_scaled_test, Y:Y_scaled_test})
    #         print(epoch,training_cost,testing_cost)
    #         training_writer.add_summary(training_summary,epoch)
    #         test_writer.add_summary(test_summary, epoch)
    saver.restore(session,"logs\\trained_model.ckpt")
    print("Training Complete")
    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_test, Y: Y_scaled_test})
    print("Final training cost: {}".format(final_training_cost))
    print("Final testing cost: {}".format(final_testing_cost))
    # save_path=saver.save(session,"logs\\trained_model.ckpt")
    Y_predicted_scaled=session.run(prediction,feed_dict={X:X_scaled_test})
    Y_predicted=Y_scale.inverse_transform(Y_predicted_scaled)

    real_earnings=testing_dataset['total_earnings'].values[2]
    predicted_earnings=Y_predicted[2][0]
    print("The actual earning==={}: where as predicted is {}".format(real_earnings,predicted_earnings))