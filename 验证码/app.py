import base64

import numpy as np
import tensorflow as tf

from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER   # 验证码字符集
CAPTCHA_LEN = 4            # 验证码长度
CAPTCHA_HEIGHT = 60        # 验证码高度
CAPTCHA_WIDTH = 160        # 验证码宽度

# 10 个 Epochs 训练的模型，因为我们使用rmsprop在10个epoch之后基本上已经要过拟合了
MODEL_FILE = './pre-trained/model/captcha_rmsprop_binary_crossentropy_bs_100_epochs_10.h5'

# 解码长度为40 的向量
def vec2text(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LEN, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text

def rgb2gray(img):
    # Y' = 0.299 R + 0.587 G + 0.114 B 
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

app = Flask(__name__) # 创建 Flask 实例

# 定义测试 的URL，如果在本地启动那么我们访问127.0.0.1:port/ping 时不论采用的是get方法还是post方法都会返回一个pong
@app.route('/ping', methods=['GET', 'POST'])
def hello_world():
    return 'pong'
# 创建一个验证码识别的访问服务 路由名字为predict ，只能使用post方法访问
@app.route('/predict', methods=['POST'])
def predict():
    # 定义resopnse success表示返回访问是否成功的结果，prediction返回验证码图片中的字符，debug调试用的
    response = {'success': False, 'prediction': '', 'debug': 'error'}
    # 首先我们设置received_image标志为False
    received_image= False
    # 如果请求方法为POST则执行下面代码
    if request.method == 'POST':
        if request.files.get('image'): # 如果我们post的数据是图像文件
            # 读取我们的图像
            image = request.files['image'].read()
            # 如果访问方法为POST，则received_image标志为True
            received_image = True
            # 然后可以在 response[debug] 添加信息 如将值设为get image
            response['debug'] = 'get image'
        elif request.get_json(): # 也有可能我们上传的图片在网络中进行了 base64 编码
            encoded_image = request.get_json()['image']
            image = base64.b64decode(encoded_image)
            received_image = True
            response['debug'] = 'get json'
        #如果 received_image标志为True则执行下面代码
        if received_image:
            # 将图像用Image.open打开然后转换成np数组
            image = np.array(Image.open(BytesIO(image)))
            # 灰度处理，并转成float32类型，然后再进行数据归一化
            image = rgb2gray(image).reshape(1, 60, 160, 1).astype('float32') / 255
            # 定义当前的会话为我们的数据流图，虽然之前没有说过在使用keras的时候定义数据流图，其实在底层keras还是会引用到session会话。
            # 然后用这个数据流图进行我们的预测，这个数据流图就是我们的模型
            with graph.as_default():
                # model.predict(image) 会在数据流图中跑一遍我们的前项神经网络 然后得到我们的pred，pred就是长度为40的1维的向量
                pred = model.predict(image)
            # vec2text(pred) 会将我们的向量转化为字符，即就是我们预测的数字，然后将字符加到我们的response['prediction']中
            response['prediction'] = vec2text(pred)
            # 之后将response['success']设置为True 表示访问成功了
            response['success'] = True
            # 将debug设置为predicted表示预测成功
            response['debug'] = 'predicted'
    else:
        response['debug'] = 'No Post'
    return jsonify(response)

model = load_model(MODEL_FILE) # 加载模型
graph = tf.get_default_graph() # 获取 TensorFlow 默认数据流图