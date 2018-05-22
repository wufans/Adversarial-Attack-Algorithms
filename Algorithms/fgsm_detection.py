"""
Dependencies: python3, tensorflow v1.4, numpy, matplotlib
"""
import os

import numpy as np

import matplotlib
#matplotlib.use('Agg')           # 设置Agg属性，可以让图形不显示，直接保存
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec #可以自定义输出图形布局的库
import keras
import tensorflow as tf
from sklearn import decomposition


from fast_gradient import fgm

"""
parameters in this code
"""
img_size = 28
img_chan = 1
n_classes = 10
pca_components = 27
random_scale = 0.5
bit_depth = 1
print("========================="*2)
print('\nLoading MNIST')
#读取mnist数据集

mnist = keras.datasets.mnist
#处理之前，X_train.shape : (60000,28,28)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print("X_train",X_train.shape)
#处理之后，X_train.shape : (60000,28,28,1)
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
#print("X_train2",X_train.shape)
#print(type(X_train))
X_train = X_train.astype(np.float32) / 255

X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255
#print("x_test",X_test[0].shape)

#将MNIST标签变成one-hot表示
to_categorical = keras.utils.to_categorical
#print("y_train",y_train[0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#print("y_train",y_train[0])

print("=========================")
print('\nSpliting data')

#随机改变array的次序
#X_train.shape:(60000,28,28,1)
# X_train.shape[0] = 60000
ind = np.random.permutation(X_train.shape[0])
#随机化X_train和y_train的顺序
X_train, y_train = X_train[ind], y_train[ind]

#从训练集中分出10%作为验证集
VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
#n = 5400
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]


print('\nConstruction graph')

"""
定义一个分类模型：
注意在这里的with语句和python中的其他地方with语句不一样，执行完毕之后conv0/和conv1/等空间还是在内存中，如果再次执行with tf.variable_scope('conv0')，就会生成名为conv0_1的空间。但是可以通过with tf.variable_scope('conv0'，reuse=True)共享变量空间
"""
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
        #tf.layers.dropout()中的training参数:Whether to return the output in training mode (apply dropout) or in inference mode (return the input untouched).
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')
    #根据ligits的值决定模型是返回最终的分类值y还是模型最后输出层的值。
    if logits:
        return y, logits_
    return y


class Dummy:
    """
    x : tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),name='x')
    y : tf.placeholder(tf.float32, (None, n_classes), name='y')
    training : tf.placeholder_with_default(False, (), name='mode')
    ybar : model(env.x, logits=True, training=env.training)
    acc : tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
    loss : tf.reduce_mean(xent, name='loss')
    train_op : optimizer.minimize(env.loss)
    saver : tf.train.Saver()
    fgsm_eps : tf.placeholder(tf.float32, (), name='fgsm_eps')  学习率
    fgsm_epochs : tf.placeholder(tf.int32, (), name='fgsm_epochs') 迭代次数
    x_fgsm : fgm(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)

    """
    pass


env = Dummy()


with tf.variable_scope('model'):
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
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=True):
    env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    env.x_fgsm = fgm(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('Evaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc




def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

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
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))


def predict(sess, env, X_data, batch_size=128):
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
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgsm, feed_dict={
            env.x: X_data[start:end],
            env.fgsm_eps: eps,
            env.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()

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
################################################################################
def spatial_smooth(x_test):

    pass
################################# 结果评价 ######################################
def result_eval(sess,env,x_test,x_adv,x_test_pre,x_adv_pre):
    """
    从TP, FN, FP, TN, P, R 6个标准评价样本
    """
    TP, FN, FP, TN = 0, 0, 0, 0
    y_test = predict(sess, env, x_test)
    y_adv = predict(sess, env, x_adv)
    y_test_pre = predict(sess, env, x_test_pre)
    y_adv_pre = predict(sess, env, x_adv_pre)

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

    return TP, FN, FP, TN, P, R
################################################################################
################################################################################
# X_test = np.reshape(X_test,[-1, img_size,img_size])
# print("-------------",X_test[0])
# plt.figure()
# plt.imshow(np.reshape(X_test[0],[img_size,img_size]),cmap="binary")
# plt.show()
# x_redu_bit = reduce_color_bits(X_test,bit_depth)
# x_redu_bit = np.reshape(x_redu_bit,[-1, img_size,img_size])
# print("-------------",x_redu_bit[0])
################################################################################
##################################### Testing ##################################
print('\nTraining')

#是否需要重新训练
train(sess, env, X_train, y_train, X_valid, y_valid, load=True, epochs=5,name='mnist')
#train(sess, env, X_train, y_train, X_valid, y_valid, load=False, epochs=5, name='mnist')

print('\nGenerating adversarial data')
X_adv = make_fgsm(sess, env, X_test, eps=0.02, epochs=12)

# print("\nReducing color bits")
# x_redu_bit = reduce_color_bits(X_test,bit_depth = 1)
# x_adv_redu_bit = reduce_color_bits(X_adv,bit_depth = 1)
# TP, FN, FP, TN, P, R = result_eval(sess,env,X_test,X_adv,x_redu_bit,x_adv_redu_bit)
# print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN,"\nP:",P,"\nR:",R)

# print('\nGenerating PCA data, n_components={0}'.format(pca_components))
# X_pca = data_pca(X_test,n_components = pca_components)
# X_adv_pca = data_pca(X_adv,n_components = pca_components)
# TP, FN, FP, TN, P, R = result_eval(sess,env,X_test,X_adv,X_pca,X_adv_pca)
# print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN,"\nP:",P,"\nR:",R)

# print('\nGenerating decenration data')
# x_decentra = decentra_data(X_test)
# X_adv_decentra = decentra_data(X_adv)
# TP, FN, FP, TN, P, R = result_eval(sess,env,X_test,X_adv,x_decentra,X_adv_decentra)
# print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN,"\nP:",P,"\nR:",R)

# print('\nGenerating randomization data')
# x_rand = random(X_test)
# X_adv_rand = random(X_adv)
# TP, FN, FP, TN, P, R = result_eval(sess,env,X_test,X_adv,x_rand,X_adv_rand)
# print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN,"\nP:",P,"\nR:",R)


# TP,FN,FP,TN = ensemble(sess, env, X_test, X_adv, X_pca, X_adv_pca, x_decentra, X_adv_decentra, x_rand, X_adv_rand)
# print("TP:",TP,"\nFN:",FN,"\nFP:",FP,"\nTN:",TN)



"""
#print('\nClean data PCA')
#X_pca = data_pca(X_test,n_components = pca_components)

#不同n的变化
X_pca = []
for n in range(img_size):
    n += 1
    print('Origin data PCA for components = {0}'.format(n))
    X_pca.append(data_pca(X_test,n_components = n))

#print('\nAdversarial data PCA')
#X_adv_pca = data_pca(X_adv,n_components = pca_components)

X_adv_pca = []#经过PCA处理的所有对抗样本的数组
for n in range(img_size):
    n += 1
    print('Adversarial data PCA for components = {0}'.format(n))
    X_adv_pca.append(data_pca(X_adv,n_components = n))


#画出四种MNIST图像
plt.figure()
plt.subplot(1,4,1)
plt.imshow(np.reshape(X_test[0],[img_size,img_size]),cmap="binary")
plt.subplot(1,4,2)
plt.imshow(np.reshape(X_pca[0],[img_size,img_size]),cmap="binary")
plt.subplot(1,4,3)
plt.imshow(np.reshape(X_adv[0],[img_size,img_size]),cmap="binary")
plt.subplot(1,4,4)
plt.imshow(np.reshape(X_adv_pca[0],[img_size,img_size]),cmap="binary")
plt.show()
"""
################################################################################

# print('\nEvaluating on clean data')
# evaluate(sess, env, X_test, y_test)
# print('\nEvaluating on randomization clean data')
# evaluate(sess, env, x_rand, y_test)
# print('\nEvaluating on adversarial data')
# evaluate(sess, env, X_adv, y_test)
# print('\nEvaluating on randomization adversarial data')
# evaluate(sess, env, X_adv_rand, y_test)


# print('\nEvaluating on clean data')
# evaluate(sess, env, X_test, y_test)
# print('\nEvaluating on decentration clean data')
# evaluate(sess, env, x_decentra, y_test)
# print('\nEvaluating on adversarial data')
# evaluate(sess, env, X_adv, y_test)
# print('\nEvaluating on decentration adversarial data')
# evaluate(sess, env, X_adv_decentra, y_test)


"""
plt.figure()
plt.subplot(1,4,1)
plt.imshow(np.reshape(X_test[0],[img_size,img_size]),cmap="binary")
plt.subplot(1,4,2)
plt.imshow(np.reshape(x_rand[0],[img_size,img_size]),cmap="binary")
plt.subplot(1,4,3)
plt.imshow(np.reshape(X_adv[0],[img_size,img_size]),cmap="binary")
plt.subplot(1,4,4)
plt.imshow(np.reshape(X_adv_rand[0],[img_size,img_size]),cmap="binary")
plt.show()

print('\nEvaluating on pca_clean data, components = {0}'.format(pca_components))
evaluate(sess, env, X_pca, y_test)

clean_pca_acc = []
for index,X_pca_singal in enumerate(X_pca):
    print('Evaluating on clean data, components = {0}'.format(index))
    loss,acc = evaluate(sess, env, X_pca_singal, y_test)
    clean_pca_acc.append(acc)

print('\nEvaluating on pca_adv data, components = {0}'.format(pca_components))
evaluate(sess, env, X_adv_pca, y_test)

adv_pca_acc = []
for index,X_adv_pca_singal in enumerate(X_adv_pca):
    print('Evaluating on pca_adv data, components = {0}'.format(index))
    loss,acc = evaluate(sess, env, X_adv_pca_singal, y_test)
    adv_pca_acc.append(acc)
"""
"""
#画出准确率随着N的变化的图像
plt.figure()
plt.subplot(1,2,1)
plt.plot(clean_pca_acc)
plt.subplot(1,2,2)
plt.plot(adv_pca_acc)
plt.show()
"""
################################################################################
"""
print('\nComparison of output between origin data and randomization data')

y_clean = predict(sess, env, X_test) #(10000,10)
y_rand = predict(sess, env, x_rand)
y_adv = predict(sess, env, X_adv)
y_adv_rand = predict(sess, env, X_adv_rand)

#分析高斯随机前后输出的变化情况
diff_clean = com_diff(y_clean,y_rand)
diff_adv = com_diff(y_adv,y_adv_rand)
print("经过高斯随机处理（方差0.4）后，真实样本分类发生改变的概率",diff_clean)
print("经过高斯随机处理（方差0.4）后，对抗样本分类发生改变的概率",diff_adv)

################################################################################

print('\nComparison of output between origin data and decentration data')

y_clean = predict(sess, env, X_test) #(10000,10)
y_decentra = predict(sess, env, x_decentra)
y_adv = predict(sess, env, X_adv)
y_adv_decentra = predict(sess, env, X_adv_decentra)

#分析去中心前后输出的变化情况
diff_clean = com_diff(y_clean,y_decentra)
diff_adv = com_diff(y_adv,y_adv_decentra)
print("经过去中心化处理后，真实样本分类发生改变的概率",diff_clean)
print("经过去中心化处理后，对抗样本分类发生改变的概率",diff_adv)

################################################################################

print('\nComparison of output between origin data and PCA data')

y_clean = predict(sess, env, X_test) #(10000,10)
y_pca = predict(sess, env, X_pca)
y_adv = predict(sess, env, X_adv)
y_adv_pca = predict(sess, env, X_adv_pca)

#分析PCA前后输出的变化情况
diff_clean = com_diff(y_clean,y_pca)
diff_adv = com_diff(y_adv,y_adv_pca)
print("经过PCA处理后，真实样本分类发生改变的概率",diff_clean,"  components =",pca_components)
print("经过PCA处理后，对抗样本分类发生改变的概率",diff_adv,"  components =",pca_components)
################################### reducing color bits #############################################
# diff_clean_all = []
# diff_adv_all = []
# for bit_depth_1 in range(1,8):
#     x_redu_bit = reduce_color_bits(X_test,bit_depth = bit_depth_1 )
#     x_adv_redu_bit = reduce_color_bits(X_adv,bit_depth = bit_depth_1)
#     y_clean = predict(sess, env, X_test) #(10000,10)
#     y_redu_bit = predict(sess, env, x_redu_bit)
#     y_adv = predict(sess, env, X_adv)
#     y_adv_redu_bit = predict(sess, env, x_adv_redu_bit)
#     diff_clean = com_diff(y_clean,y_redu_bit)
#     diff_adv = com_diff(y_adv,y_adv_redu_bit)
#     diff_clean_all.append(diff_clean)
#     diff_adv_all.append(diff_adv)
#     print("经过color reduce(bit_depth = {0})后，真实样本分类发生改变的概率".format(bit_depth_1),diff_clean)
#     print("经过color reduce(bit_depth = {0})后，对抗样本分类发生改变的概率".format(bit_depth_1),diff_adv)
# plt.figure()
# plt.plot(diff_clean_all)
# plt.plot(diff_adv_all)
# plt.show()
#画出四种MNIST图像
# plt.figure()
# plt.subplot(1,4,1)
# plt.imshow(np.reshape(X_test[0],[img_size,img_size]),cmap="binary")
# plt.subplot(1,4,2)
# plt.imshow(np.reshape(x_redu_bit[0],[img_size,img_size]),cmap="binary")
# plt.subplot(1,4,3)
# plt.imshow(np.reshape(X_adv[0],[img_size,img_size]),cmap="binary")
# plt.subplot(1,4,4)
# plt.imshow(np.reshape(x_adv_redu_bit[0],[img_size,img_size]),cmap="binary")
# plt.show()

# print("\nComparison of output between origin data and reduction color bits data")
# y_clean = predict(sess, env, X_test) #(10000,10)
# y_redu_bit = predict(sess, env, x_redu_bit)
# y_adv = predict(sess, env, X_adv)
# y_adv_redu_bit = predict(sess, env, x_adv_redu_bit)
# diff_clean = com_diff(y_clean,y_redu_bit)
# diff_adv = com_diff(y_adv,y_adv_redu_bit)
# print("经过color reduce(bit_depth = {0})后，真实样本分类发生改变的概率".format(bit_depth),diff_clean)
# print("经过color reduce(bit_depth = {0})后，对抗样本分类发生改变的概率".format(bit_depth),diff_adv)

"""

#随机抽取十个样本并打印图片
# print('\nRandomly sample adversarial data from each category')
#
# z0 = np.argmax(y_test, axis=1) #正确label
#
# z1 = np.argmax(y_clean, axis=1) #测试label
# z2 = np.argmax(y_adv, axis=1) #对抗样本label
#
# X_tmp = np.empty((10, 28, 28))
# y_tmp = np.empty((10, 10))
# for i in range(10):
#     print('Target {0}'.format(i))
#     ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
#     cur = np.random.choice(ind)
#     X_tmp[i] = np.squeeze(X_adv[cur])
#     y_tmp[i] = y_adv[cur]
#
# print('\nPlotting results',"*"*20)
#
# fig = plt.figure(figsize=(10, 1.2))
# gs = gridspec.GridSpec(1, 10, wspace=0.05, hspace=0.05)
#
# label = np.argmax(y_tmp, axis=1)
# proba = np.max(y_tmp, axis=1)
# for i in range(10):
#     ax = fig.add_subplot(gs[0, i])
#     ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]),
#                   fontsize=12)
#
# print('\nSaving figure')
#
# gs.tight_layout(fig)
# os.makedirs('img', exist_ok=True)
# plt.savefig('img/fgsm_mnist.png')
# #plt.show()
