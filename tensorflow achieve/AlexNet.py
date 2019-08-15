#载入datetime、math、time，并载入TensorFlow
from datetime import datetime
import math
import time
import tensorflow as tf

#设置batch_size为32， num_batches为100，即总共测试100个batch的数据
batch_size=32
num_batches=100

#定义用来显示网络每一层结构的函数print_actication，展示每一个卷积层或池化层输出tensor的尺寸，
#输入为tensor，显示名称t.op.name，尺寸t.get_shape.as_list()
def print_activations(t):
	print(t.op.name, ' ', t.get_shape().as_list())

#定义函数inference，接受images输入， 返回最后一层pool5及模型参数
def inference(images):
	parameters = []

	with tf.name_scope('conv1') as scope:
		#定义第一个卷积层，使用tf.truncated_normal截断的正态分布函数，标准差为0.1，初始化卷积核的参数kernel
		#卷积核尺寸为11x11，颜色通道3，卷积核数量64
		kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
		#conv2d函数完成卷积操作，步长设置为4x4
		conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
		#biases全部初始化为0，再加上conv，并使用激活函数relu进行非线性处理
		biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name=scope)
		#打印conv1，并将kernel、biases添加到parameters
		print_activations(conv1)
		parameters += [kernel, biases]

	#在第一个卷积层后添加LRN层和最大池化层
	lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
	pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
	print_activations(pool1)

	#第二个卷积层，在con1的基础上修改参数
	with tf.name_scope('conv2') as scope:
		kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope)
		parameters += [kernel, biases]
		print_activations(conv2)

	#第二卷积层后的LRN层和最大池化层
	lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
	pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
	print_activations(pool2)

	#第三卷积层
	with tf.name_scope('conv3') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(bias, name=scope)
		parameters += [kernel, biases]
		print_activations(conv3)

	#第四卷积层
	with tf.name_scope('conv4') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(bias, name=scope)
		parameters += [kernel, biases]
		print_activations(conv4)

	#第五卷积层
	with tf.name_scope('conv5') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(bias, name=scope)
		parameters += [kernel, biases]
		print_activations(conv5)

	#最后一个最大池化层
	pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
	print_activations(pool5)

	return pool5, parameters


#实现一个每轮计算时间的函数
def time_tensorflow_run(session, target, info_string):
	num_steps_burn_in = 10
	total_duration = 0.0
	total_duration_squared = 0.0

	for i in range(num_batches + num_steps_burn_in):
		start_time = time.time()
		_ = session.run(target)
		duration = time.time() - start_time
		if i >= num_steps_burn_in:
			if not i % 10 :
				print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
			total_duration += duration
			total_duration_squared += duration * duration

	mn = total_duration / num_batches 
	vr = total_duration_squared / num_batches - mn * mn 
	sd = math.sqrt(vr)
	print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))


#主函数
def run_benchmark():
	with tf.Graph().as_default():
		image_size = 224
		images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
		pool5, parameters = inference(images)

		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)

		#计算测评
		time_tensorflow_run(sess, pool5, "Forward")

		objective = tf.nn.l2_loss(pool5)
		grad = tf.gradients(objective, parameters)
		time_tensorflow_run(sess, grad, "Forward-backword")

run_benchmark()
