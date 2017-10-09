#仅对AlexNet的计算速度进行测试，不涉及训练与预测
from datetime import datetime
import math
import time
import tensorflow as tf

#测试100个banch，每个banch有32个样本
batch_size=32
num_batches=100

#定义该函数用于返回每一层计算后的shape
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

#定义该函数，建构网络，输入image图像样本，输出每一层计算后的尺寸，返回pool5计算后的结果以及网络参数集 
def inference(images):
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]


  # pool1 （pool之前先进行lrn处理）
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)

  # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

  # pool2   （pool之前先进行lrn处理）
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

  # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

  # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

  # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

  # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)

    return pool5, parameters

#定义时间评估函数，
def time_tensorflow_run(session, target, info_string):
    #头10次包含显存加载、cache命中等问题，因此用热身次数来跳过头10次 
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):#即110次
        start_time = time.time()
        #每run一次target(pool5或grad)，就得到一个duration
        session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:#i在第10次以后
            if not i % 10:#即i=10、20、30。。。、110
                print ('%s: 第 %d 步（batch）, 该步（batch）耗时 = %.3f 秒' %
                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration  #在i大于10后开始累加
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn #计算vr的目的是为了计算标准差
    sd = math.sqrt(vr)
    print ('%s: %s 经过 %d 步（batch）汇总平均, 每步（batch）用时 %.3f +/- %.3f 秒' %
           (datetime.now(), info_string, num_batches, mn, sd)) 



def run_benchmark():
    with tf.Graph().as_default():
    # 定义图像尺寸
        image_size = 224
    #自定义一个输入的图像tensor，而不是使用真实的ImageNet数据。
        images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

    # 构建计算图
        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()

    # 运行计算图，初始化变量
        sess = tf.Session()
        sess.run(init)

    # 运行计算图，求pool5，可得出前向计算的每个batch的耗费时间
        time_tensorflow_run(sess, pool5, "Forward")

    # 构建pool5的损失函数
        objective = tf.nn.l2_loss(pool5)
    # 构建loss的梯度
        grad = tf.gradients(objective, parameters)
    # 运行计算图，求梯度，可得出后向计算梯度时，每个batch的耗费时间。
        time_tensorflow_run(sess, grad, "Forward-backward")


run_benchmark()
