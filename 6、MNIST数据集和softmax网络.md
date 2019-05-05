# MNIST数据集和Softmax网络介绍


## MNIST 数据集介绍

MNIST 是一套手写体数字的图像数据集，包含 60,000 个训练样例和 10,000 个测试样例， 由纽约大学的 Yann LeCun 等人维护。
  
   

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/1.png)

## 获取 MNIST 数据集

MNIST官网链接：http://yann.lecun.com/exdb/mnist/index.html
![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/2.png)

    train-images-idx3-ubyte.gz: training set images (9912422 bytes) 训练图片
    train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)   训练标签
    t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 测试集图片
    t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)   测试集标签

## MNIST 手写体数字介绍

MNIST 图像数据集使用形如［28，28］的二阶数组来表示每个手写体数字，数组中 的每个元素对应一个像素点，即每张图像大小固定为 28x28 像素。

每一个数字图片中都是一个28x28的像素点组成，图像中的每一个像素都是由我们数组中的一个值表示的。

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/3.png)

## MNIST 手写体数字介绍

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/4.png)

手写体数字其实是256阶灰度图来表示的。

最右边的渐变图就是我们的显色卡，我们可以发现，当我们的背景趋于白色时，灰度值为0，当我们的颜色越来越深的时候，灰度值越来越大，当灰度值为255时，颜色为最深的黑色，

我们的MNIST数据集也是一样的，我们使用256阶的灰度图表示我们的手写体数字，也就表示图中的每一个像素点都存储了他的灰度值，大多的灰度值0，因为都是白色的，我们在实际的训练的时候使用灰度值本身不是一个很好的方式，我们需要把数据进行一个规范化，转为0-1之间的一个规范化的数据，并且通过我们的规范化，加速我们的训练，实际在规范化的时候，我们会将他的值除以一个255，然后我们会得到一个他们规范化之后的数据。

图片里的1，其实就是我们右边框框里面规范化后的数据，我们可以看到我们中间最深的部分其实是等于1，其实和渐变图中的255是对应起来的，矩阵中为0的数据其实就是我们渐变图中为灰度值为0的点、我们矩阵中的.7 .4 .5 .6 其实就是表示我们图中的颜色没有那么深前景。


## MNIST 手写体数字介绍

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/5.png)


由于每一张图片的尺寸都是28 x 28，为了方便连续存储，我们可以将形如[28,28]的二阶数组“平摊成”形如[784, ] (28*28)=784 的一阶数组。


(我们的灰度值是0-255,也就是256种灰度值，在我们一个28 x 28的像素图片中，我们有784个像素点，每个像素点有256种表示方法，所以一个28x28像素的图片可以组成256^784)张不同的图片。

但是这些图像并非每一张都表示有效的手写体数字，其中绝大部分都是如下的噪声图。在我们计算机中，如果我们没有给图片打上标签，那么我们的计算机是识别不出我们图片到底表示是什么数字。(这就是我们标签存在的意义)，所以我们会给我们的训练图像打上标签，表示这个训练图像表示的是几。然后在计算机上进行训练

## tf.contrib.learn 模块已被废弃

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/6.png)

## 使用 Keras 加载 MNIST 数据集

代码示例：

    tf.keras.datasets.mnist.load_data(path=‘mnist.npz’)
    Arguments:
    • path:本地缓存 MNIST 数据集(mnist.npz)的相对路径（~/.keras/datasets）
    Returns：
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/7.png)

按我们这里的代码中的方法就可以将我们的数据划分好代码中我们将我们的数据集下载到本地给他指明了路径为mnist/minist.npz (我们如果使用的操作系统为windows 那他寻找的路径为“C:\\Users\\lenovo\\.keras\\datasets\\mnist/mnist.npz”)，我们把我们的数据形状打印(print)出来就是这样的形状(比如x_train的形状是[60000, ] 由60000张图片像素是由28x28的像素组成，我们的测试集也一样形状为[10000, ] 像素为28x28 )

这里我们需要注意的是批这个概念，因为现在我们的数据集是很大的，不像之前我们的房价预测模型的数据只有47行，所以在普通的电脑上我们是很难让60000张图片一起训练的，所以我们要采用批的方式分批训练我们的模型(比如每一批32张图片，每一批64张图片等等这些都是通过我们定义超参数来实现)。


## MNIST 数据集 样例可视化

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/8.png)




## Softmax 网络介绍


## 感知机模型

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/9.png)

如何构建一个感知机模型呢？

在生物课上我们知道，我们的神经是有突触的，他帮助我们传递信息，如何传递信息呢？就是一个突触的阈值，当超过一定阈值时，我们会将消息从一个神经元传到另一个神经元，如果没有大于这个阈值，那其实我们的神经元是不会传递这个电信号的，其实在1957年代的时候就是利用这样的思想做了一个这样的模型。


我们输入神经元的信号由我们的x表示，模型参数就是我们的W项量的转置表示，xW^T就是我们的信号值，之后我们通过一个非线性的激活函数最终得到我们的输出神经元y

其中我们感知机的非线性激活函数使用的是一个简单的阈值函数，在阈值函数中，当我们的求和大于我们的阈值的时候，我们输出为1，当她小于1的时候，那我们输出0，这样我们就可以解决一个简单的二分类问题

(也就是通过我们的x和我们的权重W经过他们的线性相乘求和后，输入到我们的激活函数里面去，如果我们得到的值大于我们的激活函数中的阈值，那么我们就得到一种分类的值，如果小于我们的阈值我们就得到了另一种分类的值)


所以通过感知机的模型，我们解决了一个二分类的模型



## 神经网络

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/10.png)


在机器学习和认知科学领域，人工神经网络（ANN），简称神经网络（NN）是一种模仿生物神经网络（动物的中枢神经系统，特别是大脑）的结构和功能的数学模型或计算模型，用于对函数进行估计或近似。

比如说我们想要假设一个函数来表示我们真实世界的一个具体的问题，这个时候我们传统的方法就是找到一种表达式，比如我们的线性回归，我们希望我们的线性回归表达式s 一个单变量的或者多变量的，也可能我们的模型很复杂，无法写出表达式，但是我们有他的输入值和标签，这个时候我们把我们的数据放进神经网络中，神经网络会帮我们拟合出函数，如果拟合不出，函数也能在我们的数据集上进行工作。

神经网络就是我们在感知机上的一个扩展，神经网络是多层神经元的连接，上一层神经元的输出，作为下一层神经元的输入。


我们知道我们的感知机模型一般都是作为二分类模型，实际当我们要分类的数据很多时这个时候我们就需要将我们的感知机连接起来，做成一个多层的模型，这样就可以解决我们多分类的问题。



## 线性不可分

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/11.png)


在多分类中我们会有这样的局限，我们可能会遇到一些线性不可分的问题，比如我们想将我们图中不同颜色的点进行一个区分，第一个图我们可以直接使用简单二分类就可以做出来(如线性回归) 

那么如何处理线性不可分的情况呢？

我们可以发现在我们处理第二张图片的时候我们会遇到这样的问题，我们是无法进行线性可分的，一种方式是我们将我们的机做一种变换(先，y轴进行一些变换)，我们可以通过图三这种分线性的方程将它们区分开。还有一种方式是，我们不知道如何进行坐标轴的变化，尤其是在很复杂的模型的前提下，我们可以使用激活函数。



## 激活函数（Activation Function）

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/12.png)


当我们仅仅使用感知机的时候，我们是很难进行一个多分类模型的分类的，(很难提升他的非线性的能力)。

那我们如何提高我们的神经网络对应非线性建模的能力，我们通常会引入一些激活函数的概念，我们的激活函数都是一些非线性的函数，他可以使我们的还是呢经网络具有一些非线性的建模的能力，图中就是我们常用的一些激活函数

## 全连接层（ fully connected layers，FC ）

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/13.png)


全连接层是一种对输入数据直接做线性变换的线性计算层。它是神经网络中最常用的一种层，用于学习输出数据和输入数据之间的变换关系。全连接层可作为特征提取层使用，在学习特征的同时实现特征融合；比如我们之前讲到的，我们的特征1和特征2是独立的，我们可以通过我们的全连接层进行一个特征融合学习，一种功能是作为最终的分类层使用比如手写体识别的最红会使用一个全连接层，其输出神经元的值代表了每个输出类别的概率。  (比如我们有十个数，现在我们提取一张图片，他是6的概率很高假设有0.8，高于0.5，是其他数字的概率小于0.5，那我们在当前这个模型的最优可能的数据就是6)



## 前向传播
![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/14.png)

首先我们以一个途中的这样的三层神经网络为例，看一下前项传播到底是一个什么样的过程：

 这里先做一个符号的定义：

图中我们的网路层数就是三层(Layer1 Layer2 Layer3) w就是我们的权重，b就是我们的偏置

当我们的l=1时我们的ai(1)其实就是我们的xi，因为这个神经元输出的数据就是我们输入的数据。第二层开始我们就需要进行一个计算才能得到

-> ->
(W,B)    指我们的模型参数，W包含了很多模型的权重(而不仅仅是某一层的权重) l就是对应的层数(1-L层)，b(偏置)也是一样

__接下来我们看一下前项传播的计算过程__

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/15.png)

我们看第二层的神经元是如何得到的：

我们看第一个式子 a1(2) ，里面的计算表示 我们的第一层中的第一个神经元到第二层的第一个神经元的权重乘以x1 加上 第一层的第二个神经元到第二层第一个神经元的权重乘以x2 加上第一层的第三个神经元到第二层第一个神经元的权重乘以x3再加上第一层的第一个偏置b1 最后得到我们的机会函数a1(2)

其他的激活值(a)同理可得

第三层的输入其实和我们之前的计算是类似的，只不过他现在的输入值不是通过x得到的，因为x是我们的输入数据，x和我们第一层的模型参数组合之后经过激活得到的激活值，让激活值作为下一层的输入，然后通过类似的方式让他累计求和，在经过机会得到输出值，其实我们的Layer3和我们之前的计算是类似的，只不过我们将我们的x换成了我们的激活值a。
我们可以看出我们的Layer3是通过我们的Layer2的第一个激活值(a1(2))*Layer2的第一个神经元到Layer3的权重再 + 我们的Layer2的第二个激活值(a2(2))*Layer2的第二个神经元到Layer3的权重 +我们的Layer2的第三个激活值*Layer2的第三个神经元到Layer3的权重 +  Layer的偏置。

我们的Layer3就是我们的全连接层，他既可以输出我们的激活值，也可以作为我们分类的效果


但是我们现在这个式子看起来是十分复杂的，将式子简化后我们可以得到：

我们的简化形式就是我们的z，其实就是将我们的函数做了一个向量化的整合

## 后向传播（ Back Propagation, BP）

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/16.png)


后向传播是我们的梯度下降中的一个很重要的概念，我们的后项传播BP就是通过我们的损失函数。

我们的hx就是我们输出的预测值，当他和我们的真实值进行比较之后的差值，就是我们的误差。这个差是我们通过我们的损失函数去求得的。

那我们通过损失函数求得了这个值，我们在使用我们的最小二乘法的方式去对我们的模型参数进行求导。

我们的单变量线性回归进行求导是比较简单的，而我们的神经网络是有很多层次的，这个时候需要我们根据复合函数求导常用的“链式法则”进行求导。就像我们的a3是我们通过我们的一堆a2乘上W+b得来的。a2也是经过一堆的a1乘以w+b得来的，所以说我们对第一层进行模型求导的时候可以直接进行求解，对上一层进行模型的求导我们需要通过链式法则进行求导的，这样我们可以通过链式法则将我们的不同层次的模型参数的梯度联系起来，这样就可以使我们计算所有模型参数的梯度变得简单。

BP算法的思想早在 1960s 就被提出来了。 直到1986年， David Rumelhart 和 Geoffrey Hinton 等人发表了一篇后来成 为经典的论文，清晰地描述了BP算法的框架，才使得BP算 法真正流行起来，并带来了神经网络在80年代的辉煌。

现在TensorFlow使用的许多算法本身也是在使用梯度下降的方式进行求解。


## MNIST Softmax 网络

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86/17.png)


我们使用的网络大提分为这样几个层次

第一层输入层 我们将28x28的像素平摊为784的一维数组，神经元就对应这我们输入的图片的像素规范化后的灰度值。

中间两层使用的是各有512个神经元的隐藏层，用来学习我们的线性关系(输入数据和输出数据的关系)，每一层都有我们对应的激活函数，去做一些非线性的激活

最后我们得到一个10个神经元的全连接层，用于输出10个不同类别的“概率” 。  现在我们的数字是0-9十个数字，现在我们进行概率的判断，判断在0-9这10个标签中，哪一个标签的概率最大，(这10个值最后组合的概率应该是1)，那我们预测出来的值就是谁。

之后再通过我们的损失函数，我们的后项传播的算法去优化我们的模型的参数使得我们的模型参数越来越准确，让我们的神经网络越来越准确，识别出我们的手写体数字


这里总共涉及到了几个概念 “输入层”  “隐藏层” “全连接层”  “激活函数”   “用全连接层做分类”  的概念


















