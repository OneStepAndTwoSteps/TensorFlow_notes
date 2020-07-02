# Tensorflow notebook

## `变量管理：tf.variable_scope`

如果要在图形的各个组件之间共享一个变量，一个简单的选项是首先创建它，然后将其作为 参数传递给需要它的函数，但是当有很多共享参数时那么必须一直将它们作为参数传递，这将是非常痛苦的。

所以可以使用 tensorflow 中的 tf.variable_scope 来创建共享变量，比如：如果它还不存在，或者如果已经存在，则复用它。

### `1、创建变量`

    with tf.variable_scope("relu"):				
        threshold = tf.get_variable("threshold", shape=(),																																initializer=tf.constant_initializer(0.0))

注意：使用上面方法创建变量，`当变量已经存在时会引发异常`。这种行为可以防止错误地复用变量。

### `2、复用变量`

    with tf.variable_scope("relu", reuse=True):				

        threshold =	tf.get_variable("threshold")

注意：使用该方法创建变量，`当变量不存在时,会引发异常` 。

### `2.1、或者通过以下方法进行复用`

    with tf.variable_scope('relu') as scope:

        scope.reuse_variables()

        threshold = tf.get_variable('threshold') 
        print(tf.get_variable_scope().reuse)        # True

    print(tf.get_variable_scope().reuse)            # False

和上面一样，如果在 `with` 中设置 `reuse = True` ，那么在该 `with` 上下文中 `reuse` 都为 `True`。



### `3、详细解读`

* https://www.cnblogs.com/weizhen/p/6751792.html


## `tf.truncated_normal(shape, mean, stddev)`


*   tf.truncated_normal(shape, mean, stddev) 

### `1、用法介绍`

*   `shape` 表示生成张量的维度，`mean` 是均值，`stddev` 是标准差。

*   这个函数产生 `正态分布`，`均值` 和 `标准差` 自己设定。这是一个截断的产生 `正态分布` 的函数，就是说 __产生 `正态分布` 的值如果与 `均值` 的 `差值` 大于 `两倍的标准差` ，那就重新生成。__

*   和一般的 `正态分布` 的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。


### `2、截断正态分布`

* https://www.zhihu.com/question/49923924

## `tensorflow 中的 eval`

`eval()` 其实就是 tf.Tensor 的 Session.run() 的另外一种写法。你上面些的那个代码例子，如果稍微修改一下，加上一个 Session context manager：

        with tf.Session() as sess:
        print(accuracy.eval({x:mnist.test.images,y_: mnist.test.labels}))

其效果和下面的代码是等价的：

        with tf.Session() as sess:
        print(sess.run(accuracy, {x:mnist.test.images,y_: mnist.test.labels}))

但是要注意的是，`eval()` 只能用于 tf.Tensor 类对象，也就是有输出的 Operation。对于没有输出的 Operation, 可以用 .run() 或者 Session.run() 。Session.run() 没有这个限制。


## `tf.cast`

`tf.cast()` 函数的作用是执行 tensorflow 中张量数据类型转换，比如：


`tf.cast 结合 tf.nn.in_top_k 的使用案例：`

    import tensorflow as tf
        
    A = tf.Variable([[0.8, 0.4, 0.5, 0.6],[0.1, 0.9, 0.2, 0.4],[0.1, 0.9, 0.4, 0.2]])
    B = tf.Variable([1, 1, 2])
    result = tf.nn.in_top_k(A, B, 2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(A))
        print(sess.run(B))
        print(sess.run(result))                     # [False  True  True]

        print(sess.run(tf.cast(result,tf.float32))) # [0. 1. 1.]

上面的 in_top_k() 函数会返回一个充满布尔值的 1D 张量，因此我们 需要将这些布尔值转换为浮点数，然后计算平均值。


## `tf.assign`

语法：

        tf.assign(ref, value, validate_shape=None, use_locking=None, name=None)

函数完成了将 `value` 赋值给 `ref` 的作用。

注意：其中：`ref` 必须是 `tf.Variable` 创建的`tensor` ，如果 `ref=tf.constant()` 会报错！

案例：

        import tensorflow as tf
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


        state = tf.Variable(1)
        state_ = tf.assign(state, 10)
        with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(sess.run(state))                  # 1
        print(sess.run(state_))                 # 10
        print(sess.run(state))                  # 10


前两个计算很好懂，注意第三个 `sess.run(state)` 

这里我们能发现：上面的赋值，其实只是做了一个关于该变量点`( state --> state_ )` 的一个 `指向过程` 而已。

### `当赋值时两个 variables 类型或者维度不同：`

* 变量两个重要属性：`维度` 和 `类型` 。类型是不可改变的，但是维度在程序运行中是可能改变的，但是需要通过设置参数validate_shape=False。

* 意思就是当w1和w2两个变量的类型不一致时不能使用tf.assign(w1, w2)，但是当w1和w2两个变量的维度不一致时可以使用tf.assign(w1, w2，validate_shape=False)

## `tf.control_dependencies`

`tf.control_dependencies` ，该函数保证其辖域中的操作必须要在该函数所传递的参数中的操作完成后再进行。

        import tensorflow as tf
        a_1 = tf.Variable(1)
        b_1 = tf.Variable(2)
        update_op = tf.assign(a_1, 10)
        add = tf.add(a_1, b_1)

        a_2 = tf.Variable(1)
        b_2 = tf.Variable(2)
        update_op = tf.assign(a_2, 10)
        with tf.control_dependencies([update_op]):
        add_with_dependencies = tf.add(a_2, b_2)

        with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ans_1, ans_2 = sess.run([add, add_with_dependencies])

        print("Add: ", ans_1)                           # Add:  3
        print("Add_with_dependency: ", ans_2)           # Add_with_dependency:  12


可以看到两组加法进行的对比：

* 正常的计算图在计算 `add` 时是不会经过 `update_op` 操作的，因此在加法时 `a` 的值为 `1` 。

* 但是采用 `tf.control_dependencies` 函数，可以控制在进行 `add` 前先完成 `update_op` 的操作，因此在加法时 `a` 的值为 `10` ，因此最后两种加法的结果不同。

## `tf.GraphKeys.UPDATE_OPS`

关于 `tf.GraphKeys.UPDATE_OPS` ，这是一个 `tensorflow` 的计算图中内置的一个 `集合` ，__其中会保存一些需要在训练操作之前完成的操作__ ，并配合 `tf.control_dependencies` 函数使用。

在 `batch_normalization` 中，即为更新 `mean` 和 `variance` 的操作。也就是不通过 `tf.get_collection(tf.GraphKeys.UPDATE_OPS)` 更新操作，将无法更新 `mean` 和 `variance`

通过下面一个例子可以看到`tf.layers.batch_normalization` 中是如何实现的。



        import tensorflow as tf

        is_traing = tf.placeholder(dtype=tf.bool)
        input = tf.ones([1, 2, 2, 3])
        output = tf.layers.batch_normalization(input, training=is_traing)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print(update_ops)
        # with tf.control_dependencies(update_ops):
        # train_op = optimizer.minimize(loss)

        with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, "batch_norm_layer/Model")
        
        输出：
        [<tf.Tensor 'batch_normalization/AssignMovingAvg:0' shape=(3,) dtype=float32_ref>, <tf.Tensor 'batch_normalization/AssignMovingAvg_1:0' shape=(3,) dtype=float32_ref>]

说明：

* 可以看到输出的即为两个 `batch_normalization` 中更新 `mean` 和 `variance` 的操作，需要保证它们在 `train_op` 前完成。

* 这两个操作是在 `tensorflow` 的内部实现中自动被加入 `tf.GraphKeys.UPDATE_OPS` 这个集合的，在 `tf.contrib.layers.batch_norm` 的参数中可以看到有一项 `updates_collections` 的默认值即为 `tf.GraphKeys.UPDATE_OPS` ，而在`tf.layers.batch_normalization` 中则是直接将两个更新操作放入了上述集合。

### [`tf.control_dependencies` + `tf.GraphKeys.UPDATE_OPS`](https://blog.csdn.net/huitailangyz/article/details/85015611)

在 `批量标准化` 后进行 `梯度下降` 时可以结合使用：

        with tf.name_scope('train'):

                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                
                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                
                with tf.control_dependencies(extra_update_ops):
                        training_op = optimizer.minimize(loss)




# `tf.layers`

## `tf.layers.dense`

添加一个全连接层

## `tf.layers.batch_normalization`

BN 在如今的CNN结果中已经普遍应用，在 tensorflow 中可以通过 tf.layers.batch_normalization() 这个 op 来使用 BN 。该op隐藏了对 BN 的 mean var alpha beta 参数的显示申明，因此在训练和部署测试中需要特征注意正确使用 BN 的姿势。



# `tf.nn`

## `tf.nn.relu`

### `1、用法介绍`

*   `tf.nn.relu()` 函数是将大于 0 的数保持不变，小于 0 的数置为 0

        import tensorflow as tf
        
        a = tf.constant([-2,-1,0,2,3])
        with tf.Session() as sess:
            print(sess.run(tf.nn.relu(a)))      # [0 0 0 2 3]


## `tf.nn.in_top_k 的用法`

用法：

    tf.nn.in_top_k(predictions, targets, k, name=None)

参数：

* predictions: 你的预测结果（一般也就是你的网络输出值）大小是预测样本的数量乘以输出的维度

* target:      实际样本类别的标签，大小是样本数量的个数

* k:           每个样本中 __前K个最大的数__ 里面（序号）是否包含对应target中的值


注意：K 通常 取值为 1，表示 A 中概率最大的那个值是否等于 B 中的真实标签。

案例：

        import tensorflow as tf

        A = tf.Variable([[0.8, 0.4, 0.5, 0.6],[0.1, 0.9, 0.2, 0.4],[0.1, 0.9, 0.4, 0.2]])
        B = tf.Variable([1, 1, 2])
        result = tf.nn.in_top_k(A, B, 2)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(A))
            print(sess.run(B))
            print(sess.run(result))

        #   k=1 [False True False]
        #   k=2 [False True True]    # k = 2 时 B 中第三个元素标签为2，A中第三个向量 前两个最大值分别标签为 1 和 2，包含标签 2 所以 为True

解释：

* 这里面logits 是对分类预测结果的一个打分，每个元素对应每个可能分类的所打分数，y是真实的标签值。

* 这个函数干的是这样一件事：这个logits的打分表中前k高的分数里面，是不是包含了真实的标签值？

* 返回布尔数 True or False。



## `tf.nn.sparse_softmax_cross_entropy_with_logits`


`sparse_softmax_cross_entropy_with_logits()`	函数等同于应用	`SOFTMAX 激活函数`，然后计算 `交叉熵`，但它更高效，它妥善照顾的边界情况下，比如 `logits` 等于 0，这就是为什么我们 没有较早的应用	`SOFTMAX` 激活函数。


        with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')


我们将使 用	`sparse_softmax_cross_entropy_with_logits()`	：它根据“logit”计算交叉熵（即，在通过 `softmax`	激活函数之前的网络输出），并且期望以 0	到 -1 数量的整数形式的标签（在我们的 例子中，从 0 到 9）。 

这将给我们一个包含每个实例的交叉熵的 1D	张量。 然后，我们可以使用	`TensorFlow`	的	`reduce_mean()`	函数来计算所有实例的 `平均交叉熵`。


# `tf.train`


## `tf.train.GradientDescentOptimizer`

* tf.train.GradientDescentOptimizer() 梯度下降法	

        with tf.name_scope('train'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)              # 梯度计算
                training_op = optimizer.minimize(loss)                                  # 将计算出来的梯度应用到变量的更新中
                
## `tf.train.saver`

保存训练好的模型：

        saver = tf.train.Saver()





