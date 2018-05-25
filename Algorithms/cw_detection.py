import os
from timeit import default_timer

import numpy as np

import matplotlib
#matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras
from sklearn import decomposition

import tensorflow as tf

from cw_attack import cw


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


img_size = 28
img_chan = 1
n_classes = 10
batch_size = 32
######################################


class Timer(object):
    def __init__(self, msg='Starting.....', timer=default_timer, factor=1,
                 fmt="------- elapsed {:.4f}s --------"):
        self.timer = timer
        self.factor = factor
        self.fmt = fmt
        self.end = None
        self.msg = msg

    def __call__(self):
        """
        Return the current time
        """
        return self.timer()

    def __enter__(self):
        """
        Set the start time
        """
        #print(self.msg)
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Set the end time
        """
        self.end = self()
        #print(str(self))

    def __repr__(self):
        return self.fmt.format(self.elapsed)

    @property
    def elapsed(self):
        if self.end is None:
            # if elapsed is called in the context manager scope
            return (self() - self.start) * self.factor
        else:
            # if elapsed is called out of the context manager scope
            return (self.end - self.start) * self.factor


print('\nLoading MNIST')

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]



# It takes a while to run through the full dataset, thus, we demo the result
# through a smaller dataset.  We could actually find the best parameter
# configuration on a smaller dataset, and then apply to the full dataset.
n_sample = 10000
ind = np.random.choice(X_test.shape[0], size=n_sample, replace=False)
X_test = X_test[ind]
y_test = y_test[ind]


print('\nConstruction graph')


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        vs = tf.global_variables()
        env.train_op = optimizer.minimize(env.loss, var_list=vs)

    env.saver = tf.train.Saver()

    # Note here that the shape has to be fixed during the graph construction
    # since the internal variable depends upon the shape.
    env.x_fixed = tf.placeholder(
        tf.float32, (batch_size, img_size, img_size, img_chan),
        name='x_fixed')
    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    env.adv_train_op, env.xadv, env.noise = cw(model, env.x_fixed,
                                               y=env.adv_y, eps=env.adv_eps,
                                               optimizer=optimizer)

print('\nInitializing graph')

env.sess = tf.InteractiveSession()
env.sess.run(tf.global_variables_initializer())
env.sess.run(tf.local_variables_initializer())


def evaluate(env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = env.sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(env.sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            env.sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                                  env.y: y_data[start:end],
                                                  env.training: True})
        if X_valid is not None:
            evaluate(env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(env.sess, 'model/{}'.format(name))


def predict(env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = env.sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_cw(env, X_data, epochs=1, eps=0.1, batch_size=batch_size):
    """
    Generate adversarial via CW optimization.
    """
    print('\nMaking adversarials via CW')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        with Timer('Batch {0}/{1}   '.format(batch + 1, n_batch)):
            end = min(n_sample, (batch+1) * batch_size)
            start = end - batch_size
            feed_dict = {
                env.x_fixed: X_data[start:end],
                env.adv_eps: eps,
                env.adv_y: np.random.choice(n_classes)}

            # reset the noise before every iteration
            env.sess.run(env.noise.initializer)
            for epoch in range(epochs):
                env.sess.run(env.adv_train_op, feed_dict=feed_dict)

            xadv = env.sess.run(env.xadv, feed_dict=feed_dict)
            X_adv[start:end] = xadv

    return X_adv
################################################################################
def com_diff(y,y_pre):
    """
    PCA前后的输出变化率
    """
    z_y = np.argmax(y,axis=1)
    z_pca = np.argmax(y_pre,axis=1)
    n_samples = y.shape[0]
    diff_ori_pca = 0
    for index in range(n_samples):
        if z_y[index] != z_pca[index]:
            diff_ori_pca += 1
    diff_ori_pca /= n_samples
    return diff_ori_pca
################################################################################
def data_pca(X_data,n_components=5,zero = True,average = False,batch_size=128):
    """
    给一个28*28*1的输入，通过PCA处理

    X_test:(10000,28,28,1)

    """
    #转成(10000,28,28)
    X_data = np.reshape(X_data,[-1, img_size,img_size])
    pca = decomposition.PCA(n_components)

    n_sample = X_data.shape[0]#10000
    x_pca = np.empty_like(X_data)#tensor

    for sample in range(n_sample):
        print('pca {0}/{1}'.format(sample+1,n_sample),end='\r')
        model = pca.fit(X_data[sample])
        #print(pca.explained_variance_ratio_)
        z = model.transform(X_data[sample])
        #先降维，然后数据恢复
        ureduce = model.components_
        x_rec = np.dot(z,ureduce)
        x_pca[sample] = x_rec
        #break
    #print(x_pca.shape)
    x_pca = np.reshape(x_pca,[-1, img_size, img_size, img_chan])
    return x_pca

################################################################################
def decentra_data(X_test):
    """
    给一个10000*28*28*1的输入,各自减去平均值
    """
    X_test = np.reshape(X_test,[-1, img_size,img_size])
    n_sample = X_test.shape[0]#10000
    #print()
    x_decentra = np.empty_like(X_test)
    X_data = np.copy(X_test)
    for sample in range(n_sample):
        print('Decentration: {0}/{1}'.format(sample+1,n_sample),end='\r')
        input_pixel = np.reshape(X_data[sample],[-1])
        #print(input_pixel.shape)
        sum = 0
        #print(input_pixel)
        for pixel in input_pixel:
            sum += pixel
        average = sum /(img_size*img_size)
        #print('average',average)
        for index in range(img_size*img_size):
            input_pixel[index] -= average
        #print(input_pixel)
        #print(input_pixel.shape)
        x_decentra[sample] = np.reshape(input_pixel,[img_size,img_size])

        #break
    # print(X_test[0])
    # print('--------------')
    # print(x_decentra[0])

    x_decentra = np.reshape(x_decentra,[-1, img_size, img_size, img_chan])

    return x_decentra
################################################################################
def random(X_test,gauss = True,random_scale = 0.4):
    """
    随机高斯噪声
    """
    X_test = np.reshape(X_test,[-1, img_size,img_size])
    n_sample = X_test.shape[0]#10000
    x_random = np.empty_like(X_test)
    X_data = np.copy(X_test)
    for sample in range(n_sample):
        print('Randomization: {0}/{1}'.format(sample+1,n_sample),end='\r')
        input_pixel = np.reshape(X_data[sample],[-1])
        #sum = 0
        #print(input_pixel)
        # for pixel in input_pixel:
        #     sum += pixel
        # average = sum /(img_size*img_size)

        for index in range(img_size*img_size):
            input_pixel[index] += np.random.normal(loc=0, scale= random_scale, size=None)
        x_random[sample] = np.reshape(input_pixel,[img_size,img_size])
    x_random = np.reshape(x_random,[-1, img_size, img_size, img_chan])
    return x_random
################################################################################
def ensemble(sess,env,x_test,x_adv,x_pca,x_adv_pca,x_decentra,x_adv_decentra,x_rand,x_adv_rand):
    """
    综合利用三种处理方法检测对抗样本
    """
    y_clean = predict(sess, env, x_test)
    y_adv = predict(sess, env, x_adv)
    z_clean = np.argmax(y_clean,axis=1)
    z_adv = np.argmax(y_adv,axis=1)

    y_pca = predict(sess, env, x_pca)
    y_adv_pca = predict(sess, env, x_adv_pca)
    z_pca = np.argmax(y_pca,axis=1)
    z_adv_pca = np.argmax(y_adv_pca,axis=1)

    y_decentra = predict(sess, env, x_decentra)
    y_adv_decentra = predict(sess, env, x_adv_decentra)
    z_decentra = np.argmax(y_decentra,axis=1)
    z_adv_decentra = np.argmax(y_adv_decentra,axis=1)

    y_rand = predict(sess, env, x_rand)
    y_adv_rand = predict(sess, env, x_adv_rand)
    z_rand = np.argmax(y_rand,axis=1)
    z_adv_rand = np.argmax(y_adv_rand,axis=1)

    n_samples = y_clean.shape[0]
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for index in range(n_samples):
        if z_clean[index] == z_pca[index] and z_clean[index] == z_decentra[index] and z_clean[index] == z_rand[index]:
            TP += 1
        if z_adv[index] == z_adv_pca[index] and z_adv[index] == z_adv_decentra[index] and z_adv[index] == z_adv_rand[index]:
            FP += 1
    FN, TN = n_samples - TP, n_samples - FP

    #return TP/n_samples, FN/n_samples, FP/n_samples, TN/n_samples
    return TP,FN,FP,TN
################################################################################
def reduce_color_bits(x_test,bit_depth = 1):
    """
    减小MNIST图片的比特深度
    """
    #print("bit_depth",bit_depth)
    x_test = np.reshape(x_test,[-1,img_size,img_size])
    n_samples = x_test.shape[0]
    x_redu_bit = np.empty_like(x_test)
    x_data = np.copy(x_test)
    max_val = float(pow(2,bit_depth)-1)

    for sample in range(n_samples):
        #print('Reducing color bits: {0}/{1}'.format(sample+1,n_samples),end='\r')
        input_pixel = np.reshape(x_data[sample],[-1])
        for index in range(img_size*img_size):
            x_int = np.rint(input_pixel[index]*max_val)
            x_float = x_int/max_val
            input_pixel[index] = x_float
        x_redu_bit[sample] = np.reshape(input_pixel,[img_size,img_size])
        #break
    x_redu_bit = np.reshape(x_redu_bit,[-1, img_size, img_size, img_chan])
    return x_redu_bit
###################################### rectify #################################
def rectify(x_test,decimal = 1):
    x_test = np.reshape(x_test,[-1,img_size,img_size])
    n_samples = x_test.shape[0]
    x_rectify = np.empty_like(x_test)
    x_data = np.copy(x_test)

    for sample in range(n_samples):
        # print('Rectify bits: {0}/{1}'.format(sample+1,n_samples),end='\r')
        input_pixel = np.reshape(x_data[sample],[-1])
        for index in range(img_size*img_size):
            input_pixel[index] = round(input_pixel[index],decimal)

        x_rectify[sample] = np.reshape(input_pixel,[img_size,img_size])
        #break
    x_rectify = np.reshape(x_rectify,[-1, img_size, img_size, img_chan])
    return x_rectify
    pass
################################################################################
def spatial_smooth(x_test):

    pass
################################# 结果评价 ######################################
def result_eval(sess,env,x_test,x_adv,x_test_pre,x_adv_pre):
    """
    从TP, FN, FP, TN, P, R 6个标准评价样本
    Accuracy = (TP+TN)/(TP+FN+FP+TN)
    """
    TP, FN, FP, TN, Accuracy= 0, 0, 0, 0, 0
    y_test = predict(env, x_test)
    y_adv = predict(env, x_adv)
    y_test_pre = predict(env, x_test_pre)
    y_adv_pre = predict(env, x_adv_pre)

    #z = np.argmax(y,axis=1)
    z_test = np.argmax(y_test,axis=1)
    z_adv = np.argmax(y_adv,axis=1)
    z_test_pre = np.argmax(y_test_pre,axis=1)
    z_adv_pre = np.argmax(y_adv_pre,axis=1)

    n_samples = y_test.shape[0]

    for index in range(n_samples):
        if z_adv[index] != z_adv_pre[index]:
            TN += 1
        if z_test[index] == z_test_pre[index]:
            TP += 1

    FN, FP = n_samples - TP, n_samples - TN
    P , R = TP / (TP+FP) , TP / (TP+FN)
    Accuracy = (TP+TN)/(TP+FN+FP+TN)
    return TP, FN, FP, TN, P, R, Accuracy
################################################################################
#
#pca_components = 1
random_scale = 0.5
bit_depth = 1
decimal = 1
################################################################################
print('\nTraining')
# train(env, X_train, y_train, X_valid, y_valid, load=False, epochs=5,name='mnist')
train(env, X_train, y_train, X_valid, y_valid, load=True, epochs=5,name='mnist')

print('\nGenerating adversarial data')
X_adv = make_cw(env, X_test, eps=0.002, epochs=100)

print("\nRectifing the samples")
x_rectify = rectify(X_test,decimal = decimal)
x_adc_rectify = rectify(X_adv,decimal = decimal)
TP, FN, FP, TN, P, R = result_eval(env.sess,env,X_test,X_adv,x_rectify,x_adc_rectify)
print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN,"\nP:",P,"\nR:",R)

print("\nReducing color bits = 1")
x_redu_bit = reduce_color_bits(X_test,bit_depth = 1)
x_adv_redu_bit = reduce_color_bits(X_adv,bit_depth = 1)
TP, FN, FP, TN, P, R = result_eval(env.sess,env,X_test,X_adv,x_redu_bit,x_adv_redu_bit)
print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN,"\nP:",P,"\nR:",R)

print("\nReducing color bits = 2")
x_redu_bit = reduce_color_bits(X_test,bit_depth = 2)
x_adv_redu_bit = reduce_color_bits(X_adv,bit_depth = 2)
TP, FN, FP, TN, P, R = result_eval(env.sess,env,X_test,X_adv,x_redu_bit,x_adv_redu_bit)
print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN,"\nP:",P,"\nR:",R)

# 测试pca的维度选择
for pca_components in [1,5,10,20]:
    print('\nGenerating PCA data, n_components={0}'.format(pca_components))
    X_pca = data_pca(X_test,n_components = pca_components)
    X_adv_pca = data_pca(X_adv,n_components = pca_components)
    TP, FN, FP, TN, P, R = result_eval(env.sess,env,X_test,X_adv,X_pca,X_adv_pca)
    print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN,"\nP:",P,"\nR:",R)

print('\nGenerating decenration data')
x_decentra = decentra_data(X_test)
X_adv_decentra = decentra_data(X_adv)
TP, FN, FP, TN, P, R = result_eval(env.sess,env,X_test,X_adv,x_decentra,X_adv_decentra)
print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN,"\nP:",P,"\nR:",R)

print('\nGenerating randomization data')
x_rand = random(X_test)
X_adv_rand = random(X_adv)
TP, FN, FP, TN, P, R = result_eval(env.sess,env,X_test,X_adv,x_rand,X_adv_rand)
print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN,"\nP:",P,"\nR:",R)

