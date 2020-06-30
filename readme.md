# Tensorflow notebook

### `变量管理：tf.variable_scope`

如果要在图形的各个组件之间共享一个变量，一个简单的选项是首先创建它，然后将其作为 参数传递给需要它的函数，但是当有很多共享参数时那么必须一直将它们作为参数传递，这将是非常痛苦的。

所以可以使用 tensorflow 中的 tf.variable_scope 来创建共享变量，比如：如果它还不存在，或者如果已经存在，则复用它。

#### `创建变量`

    with tf.variable_scope("relu"):				
        threshold = tf.get_variable("threshold", shape=(),																																initializer=tf.constant_initializer(0.0))

注意：使用上面方法创建变量，`当变量已经存在时会引发异常`。这种行为可以防止错误地复用变量。

#### `复用变量`

    with tf.variable_scope("relu", reuse=True):				

        threshold =	tf.get_variable("threshold")

注意：使用该方法创建变量，`当变量不存在时,会引发异常` 。

#### `或者通过以下方法进行复用`

    with tf.variable_scope('relu') as scope:

        scope.reuse_variables()

        threshold = tf.get_variable('threshold') 
        print(tf.get_variable_scope().reuse)        # True

    print(tf.get_variable_scope().reuse)            # False

和上面一样，如果在 with 中设置 reuse = True，那么在该上下文中 reuse 都为 True。



#### `详细解读`

https://www.cnblogs.com/weizhen/p/6751792.html


