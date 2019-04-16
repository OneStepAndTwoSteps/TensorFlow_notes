# 操作和会话

__操作就是数据流图中的节点，数据流图是一种声明式的编程范式，我们通常用数据流图来表示一种算法模型。__

	TensorFlow用数据流图表示算法模型。数据流图由节点和有向边组成，每个节点均对应一个具体的操作。因此，操作是模型功能的实际载体

__数据流图中的节点按照功能不同可以分为三种__

	存储节点： 有状态的变量操作，通常用来存储模型参数。
	计算节点： 无状态的计算或控制操作，主要负责算法逻辑表达或流程控制
	数据节点： 数据的占位符操作，用于描述图外输入数据的属性

__如何获取数据流图中的数据：__

	数据流图的输入数据就是通过我们的数据节点也就是占位符从图外输入，如果想要获取图内数据，可以通过直接获取张量的数据，在会话中执行一个张量之后可以获取到张量的值，还有一种方法是执行一种特定的操作，执行之后会返回特定的张量。

在TensorFlow中我们可以理解操作的输入和输出是张量或操作(函数式编程) 因为在数据流图中一个数据的输出可以作为下一个数据的输入，就很像lambda函数,比如在数据流图中relu层的输出可以作为下一层logit层的输入，有时候一个操作的表达可以作为另一个操作的输入，以Relu layer为例，我们可以定义一个函数表达式 y=wx+b 可以看到wx矩阵相乘(Matmul)的结果可以作为下一个偏置相加(Biasadd)的输入,但是矩阵相乘输出的结果也是一个操作，而没有用单独的一个变量去定义，所以说操作的输入和输出是张量或操作本身 

__TensorFlow 典型的计算和控制操作：__

	操作类型 												典型操作
	基础算数 									add/multiply/mod/sqrt/sin/trace/fft/argmin 
	数组运算 									size/rank/split/reverse/cast/one_hot/quantize 
	梯度裁剪  									clip_by_value/clip_by_norm/clip_by_global_norm 
	逻辑控制和调试 								identity/logical_and/equal/less/is_finite/is_nan 
	数据流控制									enqueue/dequeue/size/take_grad/apply_grad/
	初始化操作									zeros_initializer/random_normal_initializer/orthogonal_initializer
	神经网络运算								convolution/pool/bias_add/softmax/dropout/erosion2d
	随机运算 									random_normal/random_shuffle/multinomial/random_gamma
	字符串运算									string_to_hash_bucket/reduce_join/substr/encode_base64
	图形处理运算 								encode_png/resize_images/rot90/hsv_to_rgb/adjust_gamma



## TensorFlow占位符操作

		TensorFlow使用占位符操作表示图外输入的数据，如训练和测试数据

		TensorFlow数据流图描述了算法模型的计算拓扑，其中的各个操作(节点)都是抽象的函数映射或数学表达式

		换句话说数据流图本身是一个具有计算拓扑和内部结构的“壳”。在用户像数据流图填充数据之前，图中没有真正执行任何计算

__如何进行数据填充:__
			
			通常利用python的数据字典，比如说

			# x=tf.placehodler(dtype,shape,name)    定义了一个占位符 placehodler中的三个参数分别对应 数据类型，形状，操作名称

			# 定义了两个正向的标量
			x=tf.placehodler(tf.int16,shape=(),name='x')
			y=tf.placehodler(tf.int16,shape=(),name='y')

			# 建立会话，然后填充数据(feed_dict={x:2,y=3})，并进行运算 (add)
			with tf.Session() as sess:
				print(sess.run(add,feed_dict={x:2,y=3}))  输出结果 5
				print(sess.run(mul,feed_dict={x:2,y=3}))  输出结果 6

			要从数据流图中取数据要通过sess.run这种方式执行，如果要向数据流图中填充数据，一般是通过占位符操作进行填充。在数据流图真正进行执行之前，数据流图其实只是一个壳，所谓的壳就是说他没有真实的数据在里面流动，此时Tensor(张量)里面什么都没有。

__注意：__
		
    我们要将TensorFlow中的python api和python区别开，因为python是一种语言，而TensorFlow中的python api只是一种领域特定语言他声明了一种声明式编程范式，所以说如果我们使用python的循环或者判断这些逻辑进行理解数据流图时，可能会出现问题，比如我们现在判断一个张量或者变量的值是否大于10，大于10的时候我们要做什么，小于10的时候去做什么，这样我们是不推荐的，因为我们在编译时刻，或者在编写代码的时刻，去预测实际运行时的状况，所以这里不推荐大家使用TensorFlow的逻辑控制




## TensorFlow会话：
	
	会话提供了估算张量和执行操作的 运行环境， 他是发放计算任务的客户端，所有计算任务都由他连接的执行引擎完成，一个会话的典型使用流程分为以下三步

__1.创建会话__

			sess=tf.Session(target=..,graph=...,config=...)           
 
			target 会话连接的执行引擎   graph 会话加载的数据流图 config 会话启动时的配置项

			traget 表示连接TensorFlow进程的执行引擎，也就是你本地的机器

			graph  会话加载的数据流图默认加载你定义的数据流图，当你同时定义多章数据流图时，这个时候你可以显式的定义你要使用哪一张数据流图

			config 比如说可以配置在使用数据流图的过程中，是否要打印设备的日志，是否打开调试信息，是否要去追踪一些数据流图的信息等等。

			可以在官网的api文档中查到具体细节。

__2.执行sess.run方法获取张量或者执行操作__

			sess.run(..)

__3.关闭会话__

			sess.close


__例子： 会话执行操作__

	import tensorflow as tf
	# 创建数据流图 z=x*y
	x=tf.placehodler(tf.float32,shape=(),name='x')
	y=tf.placehodler(tf.float32,shape=(),name='y')
	z=tf.multiply(x,y,name='z')
	sess=tf.Session()
	print(sess.run(z,feed_dict{'x':3.0,'y':2.0}))
	'''
	输出6.0
	'''


__例子：  估算张量：__
	
	注意：常量操作不需要初始化

	import tensorflow as tf

	#创建数据流图c =a+b
	a=tf.constant(1.0,name='a')
	b=tf.constant(2.0,name='b')
	c=tf.add(a,b,name='c')

	#创建会话
	sess=tf.Session()

	#估算张量c的值
	print(sess.run(c))

	'''
	输出3.0
	'''


__TensorFlow会话执行：__

	之前都是使用sess.run方法获取张量值，获取张量值还有另外两种方法分别是：估算张量 和 执行操作 

	import tensorflow as tf

	# 创建数据流图 y=Wx+b,其中W和b为存储节点，x为数据节点
	x=tf.placeholder(tf.float32)
	w=tf.Variable(1.0)
	b=tf.Variable(1.0)
	y=wx+b

	with tf.Session() as sess:
		tf.global_variable_initializer().run()  # 初始化变量 Operation.run 执行操作
		fetch = y.eval(feed_dict={x:3.0})       # 使用y.eval获取张量y的值 Tensor.eval 估算张量
		print(fetch)							# fetch = 1.0 *3.0 +1.0




__但是你如果熟悉TensorFlow的话，其实这两种获取张量值的方法，最终还是调用sess.run()，这是一种典型的反射方法__



## TensorFlow会话的执行原理：
	
	当我们调用sess.run(train_op)语句执行训练时：

	首先，程序内部提取操作依赖的所有前置操作。这些操作的节点共同组成一幅子图。

	然后，程序会将子图中的计算节点。存储节点和数据节点按照各自的执行设备分类，相同设备上的节点组成了一幅局部图。

	什么意思呢？我们要知道数据流图会在指定的设备上进行运行的，所谓的设备就是cpu，gpu，tpu等计算设备，那什么时候用cpu什么时候使用gpu呢，举个例子我们将input和reshape节点放在cpu1上，因为他们是输入的操作和形状整合的操作，然后我们把神经网络的节点放在Gpu1上，按照执行训练时的流程来看，到时候就会生成两副子图，可能cpu1是一幅图，gpu1也是一幅图，然后在真正执行的时候，可以通过可执行队列和拓扑的关系去并行的执行，如果这个节点的入度为0，就可以被放入到可执行队列，他就可以迅速的执行，每一副子图都有这样的一个可执行队列，最终就可以并行的将数据流图快速的计算完

	最后，每个设备上的局部图在执行时，根据节点间的依赖关系，将各个节点有序的加载到设备上执行。	


	可以尝试的将图中的边(张量),当他们流动到下一个节点的时候，我们将需要释放的张量给擦除掉，你可以看到每一个节点，都可以清晰的判断出他是否应该被执行，比如说input他一开始没有入边，所以他的入度为0所以他一开始就可以被执行，在执行完之后，当数据流到下一个节点后，我们将图中input流入下一个节点reshape的边擦除，这时候，reshape就称为了入度为0的节点，所以他也会开始执行，class labels也成为了入度为0的节点，所以reshape和class labels节点就可以并行的执行，在cpu1和gpu1上他们互相没有依赖，这样就可以使我们的程序快速的执行下去



__如何把节点放在单机的设备上运行呢？__

对于单机程序来说，相同机器上不同编号的cpu或gpu就是不同的设备(设备是通过不同的计算单元为单位区分的)，我们可以在创建节点时指定执行该节点的设备。

		例子：

			with tf.device("/cpu:0"):             with tf.device("/gpu:0"):  
				v=tf.Variable(...) 				  	  z=tf.matmul(x,y)

此时我们将变量放在cpu：0上执行，将矩阵相乘放在gpu：0上执行，这样当我们真正在数据流图上执行这两个操作的时候，底层的TensorFlow运行时，就可以把这两个节点放在对应的设备上并行运行，而不会造成串行的阻塞


				        (通过session会话连接到server端) sess.run传输数据流图(将传过来的数据流图根据定义划分到对应的设备上运行)	
																												-worker/cpu:0
	client：python端写的数据流图  --------------------------------------------------------------------server端- 							
																												- worker/gpu:0

__这就是为什么在python写的代码，使用python的语义在TensorFlow中有时无法被正确的运行的原因__
