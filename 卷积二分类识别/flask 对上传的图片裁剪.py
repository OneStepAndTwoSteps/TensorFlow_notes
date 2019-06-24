import base64

import numpy as np
import tensorflow as tf

from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image


def demo():
    
    return 'hello world'

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
            image = request.files['image']
            
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
        
        # **************************************************************************  对图片数据流进行resize
            
            # 对图片数据流进行resize,之后就可以传入模型进行预测
            image = Image.open(image.stream)
            image = image.resize((208, 208),Image.ANTIALIAS)  
             # 加不加都可
            #  image = np.array(image)
        
        # ************************************************************************* 
        
            # 将image传入到预测模型，得到最终的结果
            result=demo(image)
            
            # 定义返回内容
            response['prediction'] = result
            # 之后将response['success']设置为True 表示访问成功了
            response['success'] = True
            # 将debug设置为predicted表示预测成功
            response['debug'] = 'predicted'
    else:
        response['debug'] = 'No Post'
    # 使用 jsonify 返回数据
    return jsonify(response)

if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False