# TensorBoard可视化工具

## TensorBoard对我们的帮助：
  
  __一、在数据处理过程中，用户通常想要可视化地直观查看数据集分布情况。__
    
  当我们使用matplotlib进行可视化单变量和双变量线性回归时，我们是能比较方便的将数据分布的图做出来的，但是我们如果想要画出一个更高维度的图，那使用matplotlib是很难画出数据集的分布情况了。如果我们想要进行可视化，我们需要一些降维的工具，TensorBoard中自带着一些降维的工具。

  __二、在模型设计过程中，用户往往需要分析和检查数据流图是否正确实现。__

  我们有说过虽然我们的数据流图定义好了，但是我们不能确定我们定义好的数据流图是否按照我们的想法去执行，接下来我们会将如何可视化数据流图，如何查看我们的模型定义是否正确等。

  __三、在模型训练过程中，用户也常常需要关注模型参数和超参数变化趋势。__

  比如我们的θ0 θ1 θ2 这些参数的变化趋势是什么样的，在我们之前的模型中我们定义的参数很少，只有三个，但是在深度学习中我们的参数可能存在成千上万个，那我们也可以通过tensorboard来看。


  __四、在模型测试过程中，用户也往往需要查看准确率和召回率等评估指标。__

  我们在模型训练的时候也需要看 准确率和召回率这些指标，这些也可以通过tensorboard查看



  因此，TensorFlow 项目组开发了机器学习可视化工具 TensorBoard ， 它通过展示直观的图形，能够有效地辅助机器学习程序的开发者和使用者理解算法模型及其工作流程，提升模型开发工作效率。


通过tensorboard我们可以很清楚的查看我们的非常复杂数据流图，能清晰的定位出错误。所以如果我们熟悉tensorboard我们开发模型的相率会大大提升。

## TensorBoard可视化训练

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/1.png)

Tensorboard具有可视化训练的功能，在训练的过程中我们比较关注的一些值，比如我们的损失值，这里我们通过我们的交叉熵(cross entropy) 作为我们的损失函数,这里的交叉熵我们可以看到，通过我们的x轴越来越大，我们的交叉熵的值越来越小，这也是符合我们的预期的，但是在这个图像中有两种颜色，分别是橙色和蓝色。

__黄色__ 的是我们的train 训练数据(黑色框里面有写)，我们可以看到我们的训练数据逐渐呈下降的趋势，这里的深橙和浅橙分别表拟合后的数据和我们真实的数据，在图的左侧有一个smoothing的一个设置，表示tensorboard需不需要拟合出我们的数据，当smoothing设置的足够大时，他将会拟合出一条曲线，当足够小的时候，他会以连线图的形式展现出来，这可以方便我们查看我们的数据拟合后的结果。

__蓝色__ 的线条表示我们的evel，也就是我们模型评估的数据，我们模型评估的数据当然会表现的更好，因为此时我们的模型已经训练好了，我们可以看出我们的线条出现的震荡很小，就能保证我们的模型更加的稳定。

当我们将我们的鼠标放置在我们的tensorboard图中的时候，会出现一个点，就像图中的那样，此时会显示出这个点在tensorboard中的数据，我们黑色框中的两行数据就是点中的信息。

除了我们可以显示出我们的交叉熵之外，我们可以查看我们的准确值(accuracy)的图，下面的mean图等等。

## TensorBoard可视化统计数据

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/2.png)

除了可以可视化我们的训练数据，我们还可以可视化我们的统计数据，比如以我们的softmax layer这层神经网络层为例，其实这张图就是向我们展示了softmax的权重，在我们测试集上的一个表现。

我们可以看到，在屏幕的里面y轴的数据其实表明的是我们训练的步数，从50步到1000步，x轴是我们的值的分布情况，我们可以看到我们刚开始设置的权重值是一个统一的值(从最高点可以看出，因为最高点我们的步数为0，就是刚开始的点)，随着训练步数的增加，权重值的分布也越来越趋近一个正态分布，这个就是我们可以通过我们的tensorboard很好的进行我们的参数可视化，

图中我们是一个训练集的表现

和我们上面说的测试集一样，我们可以看到我们训练集中刚开始设置的权重值是一个统一的值(从最高点可以看出，因为最高点我们的步数为0，就是刚开始的点)，随着训练步数的增加，权重值的分布也越来越趋近一个正态分布

## TensorBoard可视化数据分布

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/3.png)


我们还可以通过tensorboard查看我们的数据分布。

比如我们ppt左边的三个数据分布，分别是 伽马分布，均匀分布，和陀松分布。


同时tensorboard还可以实现一个图合并的功能，我们可以将我们的这三个分布整合到一个分布图中，就是我们右边这个大图。好处是我们每一次迭代后的结果都可以展示在图上，并且可以显示出在那一步的迭代中的特定参数的分布。



## TensorBoard可视化数据集（MNIST）


![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/4.png)


同时我们还可以使用我们的tensorboard展示我们的训练集。


比如以数据集（MNIST）为例，数据集（MNIST）是一个手写体数字的训练集，我们可以从左上部分看到，我们可以看到一个label(标签)，其中显示出来我们每一个数据他有多少张图片，如0这张图片有980张，1这张图片有1135张。

在一个高维数据中我们是无法轻易的看出他们不同数据间的分类情况，所以通常我们如果要查看一些高维的数据就会通过一些降维的方法，如tensorboard中的T-SNE(图的左下位置)就可以帮助我们将高维数据转化为我们的3维数据，是一个很好的降维工具。


图中我们可以看到我们已经进行了1424次迭代，随着时间的增加，我们迭代的次数也会增加，最后迭代出来的效果也会更好，我们可以看到降维之后相同手写体的图片基本都会聚到相同的一个位置。

通过tensorboard我们可以查看我们的数据集，即使是高维的数据集，我们也可以通过tensorboard将其进行可视化展示。

## TensorBoard可视化数据流图

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/5.png)


这里是一个可视化神经网络的数据流图 。

我们现在只是给大家先看一下可视化数据流图之后可以如何进行一些内容的查看。

## TensorBoard 使用流程

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/6.png)

可视化的数据是数据流图和张量，它们需要在会话中加载或执行操作后才能获取。然后， 用户需要使用 __FileWriter 实例__ 将这些数据写入事件文件。最后，启动 TensorBoard 程序， 加载事件文件中的序列化数据，从而可以在各个面板中展示对应的可视化对象。

如何使用 tensorboard 其实是有一个流程的，tensorboard加载的内容其实就是我们的数据流图，也就是session中的graph，还有一个就是我们的张量，我们很多会话在执行一些操作后输出的值，这些值都是我们通过在会话中加载或者通过一些操作后得到的，得到数据之后，我们通过使用我们的filewriter这样一个实例，从我们的会话中获取到，然后加入到我们的事件文件(event file)中，写入事件文件(event file)之后，就相当于保存到了我们的硬盘上，当我们持久化我们的数据之后，我们可以通过tensorboard加载事件文件(event file)，然后再进行一个可视化，所以tensorboard的数据来源其实就是这些event file，就是TensorFlow本身定义的一些序列化的数据，这些数据通过会话传入到filewriter这个实例，然后写入到event file中的，会话就是我们之前说的执行环境 我们的会话分为两类 一类是我们的sess.graph也就是我们的数据流图本身，还有一类是我们加载了一个数据流图，然后在数据流图中做了一些操作，把数据导入之后得到的实际的值，这些值通过Summary Ops汇总操作来进行记录，记录好之后，就可以通过filewriter写到我们的eventfile中了。



## tf.summary 模块介绍

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/7.png)



这里简单介绍一下TensorFlow的summary模块。

前面说过前述流程中使用的 FileWriter 实例和汇总操作（Summary Ops）其实这两个操作均属于 tf.summary 模块。 __其主要功能是获取和输出模型相关序列化数据__ ，它贯通 TensorBoard 的整个使用流程。

__tf.summary 模块__ 的核心部分由 一组汇总的操作 和三个部分组成，汇总操作它其实就涵盖了不同的可视化的数据类型，比如 audio(音频) image(图形)  scalar(标量) histogram(直方图) 等等，将这所有的操作汇总成一个操作，这一个操作就记录了所有的数据输出的结果，就可以然后就可以将其写到filewriter里面。

__tf.summary 模块__ 的核心部分由一组汇总操作以及 FileWriter、Summary 和 Event 3个类组成。

Filewriter就是我们写文件的实例，他本身就是一个writer，就和我们python 的writer一样 ，他特定作用域TensorFlow的tensorboard，然后他可以把我们的流图和我们summary opsz这些执行后的结果都写到我们的eventfile中，eventfile就是我们的Event。


## 名字作用域与抽象节点
![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/8.png)

之前我们在使用数据流图的时候是没有使用 __tf.name_scope__ 这样一个方法的，为什么要使用这样的一个方法也非常明确， __它通过一个层级的概念把我们的一些固定的节点包含下去。__

举个例子，之前在看数据流图的动画的时候我们可以发现我们的数据流图是可以展开的， __我们的input loss这些节点，其实都是可以进行点开的，点开之后里面是有更加丰富的层次结构的，使用 tf.name_scope这个名字作用域其实和我们经典的编程语言中的名字作用域一样，他具有层级的结构，当我们使用我们的tf.name_scope这个上下文语句包含我们的数据流图节点的时候，他旗下的这些数据流图都会被囊括在这个抽象节点当中。__ 而input这个节点本身其不是一个特定的操作，而是一个抽象节点的概念，这些抽象节点的作用就是帮我们去梳理数据流图的结构。使得我们的数据流图更加有层次化。

__以这三行代码为例 x输入值和y输出值 都被囊括在input这个抽象节点中了__

    with tf.name_scope('input'):
        # 输入 X，形状[47, 3]
        X = tf.placeholder(tf.float32, X_data.shape, name='X')
        # 输出 y，形状[47, 1]
        y = tf.placeholder(tf.float32, y_data.shape, name='y’)

在使用我们的tensorboard之后我们可以点开我们的input抽象节点，我们可以看到x(输入值)和y(输出值)。Hypothesis中存放了我们的权重变量和我们的推理值(y_pred) loss抽象节点中的数据不仅有从hypothesis过来的也有从我们的input抽象节点中过来的数据，所以会有边从input指向我们的loss，和hypothesis指向loss。

with tf.name_scope(‘train‘):中其实定义的就是我们的优化器，他的输入是我们的代价函数(loss) 同时也需要我们的input 因为在我们进行训练的同时我们也需要进行数据的填充，同时我们也需要将我们的模型参数(weight)传进去，因为我们要迭代我们的模型参数。

__所以说通过抽象节点和名字作用域我们可以更清晰的可视化我们的数据流图，这样我们的有向边可能更明确的表明两个节点之间的关系。__

通过这幅图我们可以看到：

我们的权重+我们的输入数据共同的构成了我们的hypothesis，，同时我们的hypothesis(假设函数)和我们的input中的输入数据一起得到了我们的loss值。我们的代价函数通过我们的优化器(train)并且传入我们的input和weight 不断的进行求最小化。



## 创建 FileWriter 实例

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/9.png)


和我们之前讲的创建会话代码不同之处在于这里我们多了一行创建filewriter的实例：

 __创建FileWriter实例，并传入当前会话加载的数据流图 当前会话也就是我们的sess.graph 同时我们传入一个路径，这个路径表示我们要把我们的filewriter创建在哪个目录下__
 
代码中写的目录是当前目录下的summary子目录下的linear-regression-1目录中。 我们的事件文件和我们的checkpoint文件都会存储在该目录下，此时我们通过filewriter就可以将我们的数据写入到eventfile了。

    writer = tf.summary.FileWriter('./summary/linear-regression-1', sess.graph)

再整一个会话执行完之后，我们需要关闭我们的filewriter输出流，这也是一个典型的数据IO的一个writer的执行操作。

__关闭FileWriter的输出流__

    writer.close() 

当我们执行成功之后会自动的帮我们将数据流图写到我们定义的这个目录的下面，也就是底下的那个图片。

也就是我们的 notebook-examples/chapter-4/summary/linear-regression-1 目录下的events.out.tfevents.1547908589.Django.lan 这个文件。这个文件可以通过我们的tensorboard进行一个可视化展示的。



## 启动 TensorBoard

  
![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/10.png)
  
Tensorboard的启动方法很简单 命令就是 tensorboard --logdir ./ --host localhost  第一个参数  --logdir就是我们日志文件夹也就是我们的eventfile文件所在的目录，--host localhost 表示我们启动tensorboard的地方是我们的本机，默认端口是6006，如果端口号已被占用，我们可以通 --port来指明我们的端口号


在tensorboard中是支持我们的层次化结构的，什么意思呢？ 就是比如我们在一个文件夹下面存在多个存储着eventfile的文件夹时，他也是可以都识别出来的同时可以使用其他参数进行过滤显示。


启动之后我们访问该链接既可  图中链接为 http://localhost:6006 

  
  
![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/11.png)  

__此时我们指定的logdir下有两个文件夹，并且这两个文件夹都存在eventfile  这里我们可以进行指定我们显示哪一个文件夹下面的数据流图。__


__回顾：__

创建一个可视化数据流图的流程：

1.创建数据流图(不一定需要使用tf.name_scope创建名称，但是如果不命名则会显得凌乱，加了name_scope可以使我们更加清晰的看我们数据流图的层次结构)

2.创建filewriter实例，我们需要传入两个参数： 第一个参数为 我们要将我们的eventfile保存在哪一个目录下，第二个参数为，我们当前会话加载的数据流图为哪一个，你传进去的数据流图也就会写到你指定的目录下。程序执行完成之后推出filewriter的输出流

3.启动tensorboard命令就是 tensorboard --logdir ./ --host localhost  第一个参数  --logdir就是我们日志文件夹也就是我们的eventfile文件所在的目录，--host localhost 表示我们启动tensorboard的地方是我们的本机，默认端口是6006，如果端口号已被占用，我们可以通 --port来指明我们的端口号


## 启动 TensorBoard (补充)

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/12.png) 

__如果启动失败__

__可以修改你虚拟环境venv下的 xxxxx\site-packages\tensorboard\manager.py文件 ，改成图中这样既可__

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/tensorboard/13.png) 






