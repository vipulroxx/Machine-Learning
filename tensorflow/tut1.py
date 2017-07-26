import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

#Just additions
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)
sess = tf.Session()
node3 = tf.add(node1,node2)
print 'Node 1 = {0} \nNode 2 = {1} \nAdding Nodes = {2}'.format(sess.run(node1) ,sess.run(node2),sess.run(node3))

#Using placeholders
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
print 'Adding nodes using placeholder when a  = {0} and b = {1} is = {2} and when a = {3} and b = {4} and multiplied by 3  is = {5}'.format(3,5,sess.run((a+b), {a:3,b:5}),[1,5],[2,4],sess.run((a+b)*3, {a:[1,5],b:[2,4]}))

#Using variable
w = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w * x + b
init = tf.global_variables_initializer()
sess.run(init)
print 'X = 1,2,3,4 our linear model has y = mx + c as {} '.format(sess.run(linear_model, {x:[1,2,3,4]}))

#Loss
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print 'loss value: {}'.format(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

#Improve the loss by changing w and b to -1 and 1
fixw = tf.assign(w, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixw,fixb])
print 'new loss : {}'.format(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

