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













