
### CNN 简介

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/CNN%E7%BD%91%E7%BB%9C/1.png)

__CNN模型是一种以卷积为核心的前馈神经网络模型。__ 20世纪60年代，Hubel和Wiesel在研究猫脑 皮层中用于局部敏感和方向选择的神经元时发现其独特的网络结构可以有效地降低反馈神经网 络的复杂性，继而提出了卷积神经网络（ConvoluTIonal Neural Networks，简称CNN）。

和softmax神经网络模型也是类似的，CNN其实在我们大脑里去对某一些特定的图像，其实是有一些神经元对他进行识别的，图中就是一个cnn模型， 图中的input和output是我们的输入输出，整个网络结构中我们可以看到上面的橙色的部分是我们的卷积层，(对应我们左下图的标签)  我们可以看到其中引入了我们的两种池化，一个是平均池化(AvgPool),一个是最大池化(MaxPool)这两个池化我们后面会讲,还有我们的Concat(将几个不同的结果连接在一起)，还有Dropout(解决过拟合的问题)，Fully connected是我们的全连接层，主要是用来你和一些函数，softmax是用来在我们输出的时候作为一个多分类器，他可以输出不同类别的概率。



### 卷积（Convolution）
  
首先我们先讲一下卷积的概念， __卷积是分析数学中的一种基础运算__ ， 其中对输入数据做运算时所用到的函数称为 __卷积核。__
  
     设:f(x), g(x)是R上的两个可积函数，作积分：我们可以写出他的表达式

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/CNN%E7%BD%91%E7%BB%9C/2.png)

可以证明，关于几乎所有的实数x，上述积分是存在的。这样，随着x的不同取值，这个积分就 定义了一个如下的新函数，称为函数f与g的卷积

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/CNN%E7%BD%91%E7%BB%9C/3.png)
![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/CNN%E7%BD%91%E7%BB%9C/3-2.png)

可以证明，关于几乎所有的实数x，上述积分是存在的。这样，随着x的不同取值，这个积分就 定义了一个如下的新函数，称为函数f与g的卷积，也就是我们的函数f和g的卷积运算

我们可以从标签中看出图中的蓝色的框是我们的f函数，红色的框是我们的g函数，他们经过卷积之后的函数就是我们的黑色线条的这个函数，我们可以看到经过卷积之后的取值模式是一种比较明确的形式，很多人认为卷积可用来进行一些翻转平移，然后做各种各样的运算，其实在数学的角度上看的话它就是一个明确的计算方法，他的作用可能在我们的卷积层中会有一个更好的体现。

### 卷积层（Convolutional Layer, conv）
  
![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/CNN%E7%BD%91%E7%BB%9C/4.png)
  
首先卷积层是使用一系列卷积核与多通道输入数据做卷积的线性计算层卷积层的提出是为了利用输入数据（如图像）中特征的局域性和位置无关性来降低整个模型的参数量。卷积运算过程与 图像处理算法中常用的空间滤波是类似的。比如求图像边缘的滤波器，求各个角度的滤波器，比如求90度180度这些角度的滤波器，其实和我们的卷积核有异曲同工之妙的，区别就是传统的图像中的一些经典的滤波器我们已经知道如何去设置我们的滤波器，但是在卷积神经网络里面，它得滤波器的参数可能是不确定的，需要我们不断的去训练，才能得到一个准确的卷积核， __因此，卷积常常被通俗地理解为一种“滤波”过程， 卷积核与输入数据作用之后得到了“滤波”后的图像(滤波后的图像通常称之为feature)，从而提取出了图像的特征(feature)。__

通过我们左下角的图，我们可以看到我们每一次取都会取出一个feature，做后这些图像组合后的图像我们称为feature map，feature map也就是我们下一个网络层的输入，我们可以看到的是在我们图像的左下角最左边的图是我们的原图，原图通过我们中间的卷积核运算之后，我们就得到了这个图像的feature，这些feature共同组成了右边这张图片(也就是feature map)

通过我们右边的手写数字体为例，右边这些图就是不同的卷积对我们的手写体数字图片进行特征提取之后得到的feature map，在第一次卷积之后得到的feature map，大家可以看到我们经过第一次卷积之后得到的feature 我们是不太知道他能提取出一些什么东西的，但是我们在经过两次卷积之后，我们得到我们的右下角的第四张图(经过三次卷积之后)我们可以发现已经具备了图形的形状了。


__这就是我们通过卷积操作提取出来的高纬度的特征，通过不断的卷积我们可以让神经网络学习到图像的特征，比如在一些更复杂的图像里面我们的图像识别里面，一开始我们可以通过卷积学到一些简单的图形(边缘图形，比如三角形啊，圆形啊等等特征)再后来我们通过卷积池化这些网络做配合之后，我们可以学习到一些比如猫脸的局部特征，这些都是可以通过卷积做到。__

同时也可以利用我们的位置的无关性和一些局部特征很好的去做模型减少的一个操作，就不需要有那么多模型参数，比如我们的MNIST softmax里面有一个784维的一阶向量(我们的手写体数字的像素28 * 28)，那么就有我们256的784次方这么一个巨大的特征空间(因为灰度图可以取256个值，如果是RGB的图像那么值就是256^3了)，当我们的像素达到300 * 300的时候我们的特征空间就已经非常巨大了，同时我们是不希望在这么大的特征空间中做运算的。 __那么卷积就可以有效的降低我们模型的参数量，他把一些无效的特征（噪声）就不再关注。那最后我们输入到网络中的feature map 就会比较小，以我们的MNIST数据集为例，我们可以把模型从784维降低为几十维的向量。__


### 池化层（Pooling）
![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/CNN%E7%BD%91%E7%BB%9C/5.png)

__池化层主要用来用于缩小数据规模的一种非线性计算层为了降低特征维度，__ 我们需要对输入数据进 行采样，具体做法是在一个或者多个卷积层后增加一个池化层。
           
__池化层由三个参数决定：__

    （1） 池化类型，一般有最大池化和平均池化两种 (平均池化(AvgPool),最大池化(MaxPool))；

    （2）池化核的大小k；

    （3）池化核的滑动间隔s

上图给出了一种的池化层示例。其中，2x2大小的池化窗口以2个单位距离在输入数据上滑动。 在池化层中， __如果采用最大池化类型，则输出为输入窗口内四个值的最大值；__

比如说原图中我们有一个2x2大小的窗口，那么这2x2的这个窗口就会对应我们输出数据中的第一个值(也就是经过池化我们将4个数据变成了一个数据)

那么这个2x2的池化窗口从输入数据到输出数据到底取什么值呢？那么如果使用我们的MaxPool(最大池化)就会取最大的值，如采用平均池化类型，则输出为输入窗口内四个值的平均值。

__这样我们的图像就从一个6x6大小的矩阵变成了一个3x3大小的矩阵，为什么这样呢，因为我们这里指定的池化核的滑动间隔为2个单位的距离，所以我们第一次会取输入数据的2x2的数据，滑动之后我们就取中间这四个数据，中间这个数据也就是对应我们输出数据的中间这个数，这样滑动之后就得到了我们的3x3的输出数据。__



### Dropout 层
![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/CNN%E7%BD%91%E7%BB%9C/6.png)

__Dropout 是常用的一种正则化方法，Dropout层是一种正则化层。__ 全连接层参数量非常庞大（占 据了CNN模型参数量的80%～90%左右），发生过拟合问题的风险比较高，所以我们通常需要 一些正则化方法训练带有全连接层的CNN模型。 __在每次迭代训练时，将神经元以一定的概率值暂时随机丢弃，即在当前迭代中不参与训练。__

__Dropout什么意思呢？__

因为我们的参数很多，在每次迭代训练的时候我们并不是希望神一个神经元都是被激活的，我们曾经通过激活函数(如果激活值小鱼阈值则不会传递到下一层)，现在我们也可以通过dorpout来做这样的事情，他就是以一定的概率值随机丢弃一些神经元，单这些丢弃的神经元只是在这一次迭代中不参与训练，下一次训练我们还可以通过Dorpout在进行丢弃。

这样一来，我们这次训练的时候就不是所有的神经元都参与训练。


以我们的图为例，我们可以看到输入输出神经元都有四个，如果我们使用dropout，那么我们输入神经元的第一个和第三个神经元就不在参与训练，这样使得我们的模型变得简单，减少了模型的参数。降低复杂度。同时随机的神经元进行组合，减少了神经元之间可能形成的共同依赖，dropout的神经网络是由dropout之后的子模型组成的，因为我们在训练完之后，我们是需要所有的神经元参与的(我们在训练的时候训练很多子模型，但是最后我们要将训练好的子模型进行一个组合，因为我们是需要所有的神经元参与预测) __这样就有利于提高我们模型的泛化能力__

### Flatten

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/CNN%E7%BD%91%E7%BB%9C/7.png)


__Flatten 就是将我们的feature map 摊平__

现在我们将图片做了特征提取，然后进行池化，得到feature map 之后，将我们的feature map 摊平，输入到我们的全连接层，(之前我们讲softmax的时候，是将我们的数据集特征化，然后摊平([28,28] -> [784, ]) 然后将输入输入全连接层) 

最后我们通过softmax函数得到一个多分类的效果，最后softmax输出10个多分类的概率，这样我们就实现了一个手写体的识别

__我们的softmax和我们的cnn网络的主要区别：__

    在输入全连接网络之前的处理，一个是输入原图，一个是输入我们卷积之后的特征。


### MNIST CNN 示意图

![Image_text](https://raw.githubusercontent.com/OneStepAndTwoSteps/TensorFlow_notes/master/static/CNN%E7%BD%91%E7%BB%9C/8.png)
这是一个我们cnn网络的示意图

我们可以看到左边是我们的输入，经过卷积之后我们得到了feature maps 经过池化，我们的feature maps 在变小，变小之后，我们在通过卷积提取他的特征，输入到全连接网络，经过全连接网络的学习之后，我们得到了高维的函数，最后通过我们的softmax得到了一个分类器，最后在我们的输出层输出概率。概率最高的值就是我们的分类结果

__Softmax 和我们的 CNN 主要的区别就是我们的CNN有一个特征提取的部分，__ 后面的流程其实都是类似的。




