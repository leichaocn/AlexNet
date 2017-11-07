# region 加载库和参数配置
#仅对AlexNet的计算速度进行测试，不涉及训练与预测
from datetime import datetime
import math
import time
import tensorflow as tf

#测试100个banch，每个banch有32个样本
batch_size=32
num_batches=30
# endregion

# region 计算图构建函数

# 打印每层输出tensor的shape
# 非常值得收藏的函数!与with tf.name_scope('conv1') as scope配合,可以查看每层结构.
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

# 计算图构建核心函数
# 定义该函数，建构网络，输入image图像样本，输出每一层计算后的尺寸，返回pool5计算后的结果以及网络参数集
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

    # 正品AlexNet到这里,还会再接两个4096输出的全连接和一个1000输出的softmax层.
    # 也要构建真实的loss和优化器.但这里省略了.

    # 计算赝品损失函数
    # tf.nn.l2_loss(var)原本是用来返回var的l2_loss,再乘以一个因子,作为var的正则损失添加到loss集合中.作为最后的loss一份子参与优化.
    objective = tf.nn.l2_loss(pool5)
    # 计算梯度
    # 后向计算耗时包括计算loss,计算梯度,梯度更新.其中,梯度更新计算量较小,因此忽略.
    grad = tf.gradients(objective, parameters)

    return pool5, grad
# endregion

# region 执行函数
# target是pool5,就是执行预测(或测试)的前向计算;target是梯度,就是执行训练的后向计算.
# 仅仅是评估时间,因此没有执行优化器.
def time_tensorflow_run(session, target, info_string):
    #头10次包含显存加载、cache命中等问题，因此用热身次数来跳过头10次 
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):#即110次
        start_time = time.time()
        #_ = 每run一次pool5，就得到一个duration
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
# endregion

def run_benchmark():
    with tf.Graph().as_default():

        # region 数据预处理
        # 定义图像尺寸
        image_size = 224
        # 自定义一个输入的图像tensor，而不是使用真实的ImageNet数据。
        images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))
        # endregion

        # region  构建计算图
        pool5, grad = inference(images)
        # 定义变量初始化op
        init = tf.global_variables_initializer()
        # endregion

        # region 执行计算图
        # 初始化变量
        sess = tf.Session()
        sess.run(init)
        # 预测.或曰测试,就是前向计算.
        # 求pool5，可得出前向计算的每个batch的耗费时间
        time_tensorflow_run(sess, pool5, "Forward")
        # 训练.即求梯度,更新权重(忽略),就是后向计算
        # 求梯度，可得出后向计算梯度时，每个batch的耗费时间。
        time_tensorflow_run(sess, grad, "Forward-backward")
        # endregion

run_benchmark()
