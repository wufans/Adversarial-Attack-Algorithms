import tensorflow as tf


__all__ = ['cw']


def cw(model, x, y=None, eps=1.0, ord_=2, T=2,
       optimizer=tf.train.AdamOptimizer(learning_rate=0.1), alpha=0.9,
       min_prob=0, clip=(0.0, 1.0)):
    
    xshape = x.get_shape().as_list()
    noise = tf.get_variable('noise', xshape, tf.float32,
                            initializer=tf.initializers.zeros)

    # scale input to (0, 1)
    x_scaled = (x - clip[0]) / (clip[1] - clip[0])

    # change to sigmoid-space, clip to avoid overflow.
    z = tf.clip_by_value(x_scaled, 1e-8, 1-1e-8)
    xinv = tf.log(z / (1 - z)) / T

    # add noise in sigmoid-space and map back to input domain
    xadv = tf.sigmoid(T * (xinv + noise))
    xadv = xadv * (clip[1] - clip[0]) + clip[0]

    ybar, logits = model(xadv, logits=True)
    ydim = ybar.get_shape().as_list()[1]

    if y is not None:
        y = tf.cond(tf.equal(tf.rank(y), 0),
                    lambda: tf.fill([xshape[0]], y),
                    lambda: tf.identity(y))
    else:
        # we set target to the least-likely label
        y = tf.argmin(ybar, axis=1, output_type=tf.int32)

    mask = tf.one_hot(y, ydim, on_value=0.0, off_value=float('inf'))
    yt = tf.reduce_max(logits - mask, axis=1)
    yo = tf.reduce_max(logits, axis=1)

    # encourage to classify to a wrong category
    loss0 = tf.nn.relu(yo - yt + min_prob)

    axis = list(range(1, len(xshape)))
    ord_ = float(ord_)

    # make sure the adversarial images are visually close
    if 2 == ord_:
        # CW-L2 Original paper uses the reduce_sum version.  These two
        # implementation does not differ much.

        # loss1 = tf.reduce_sum(tf.square(xadv-x), axis=axis)
        loss1 = tf.reduce_mean(tf.square(xadv-x))
    else:
        # CW-Linf
        tau0 = tf.fill([xshape[0]] + [1]*len(axis), clip[1])
        tau = tf.get_variable('cw8-noise-upperbound', dtype=tf.float32,
                              initializer=tau0, trainable=False)
        diff = xadv - x - tau

        # if all values are smaller than the upper bound value tau, we reduce
        # this value via tau*0.9 to make sure L-inf does not get stuck.
        tau = alpha * tf.to_float(tf.reduce_all(diff < 0, axis=axis))
        loss1 = tf.nn.relu(tf.reduce_sum(diff, axis=axis))

    loss = eps*loss0 + loss1
    train_op = optimizer.minimize(loss, var_list=[noise])

    # We may need to update tau after each iteration.  Refer to the CW-Linf
    # section in the original paper.
    if 2 != ord_:
        train_op = tf.group(train_op, tau)

    return train_op, xadv, noise
