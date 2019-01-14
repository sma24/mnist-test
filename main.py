import tensorflow as tf
import numpy as np
from flask import Flask,request
from PIL import Image
from io import BytesIO
from scipy import misc
from mnist import model

def model_ini():
    sess = tf.Session()
    saver = tf.train.import_meta_graph("./data/convolutional.ckpt.meta")
    saver.restore(sess,tf.train.latest_checkpoint('./data'))
    return sess

def Number_recognition(file,sess):
    im = Image.open(BytesIO(file))
    imout=im.convert('L')
    xsize, ysize=im.size
    if xsize != 28 or ysize!=28:
        imout=imout.resize((28,28),Image.ANTIALIAS)
    arr = []
    for i in range(28):
        for j in range(28):
            pixel = float(1.0 - float(imout.getpixel((j, i)))/255.0)
            arr.append(pixel)
    keep_prob=tf.get_default_graph().get_tensor_by_name('dropout/Placeholder:0')
    x=tf.get_default_graph().get_tensor_by_name('x:0')
    y=tf.get_default_graph().get_tensor_by_name('fc2/add:0')
    arr1 = np.array(arr).reshape((1,28*28))
    pre_vec=sess.run(y,feed_dict={x:arr1,keep_prob:1.0})
    pre =str(np.argmax(pre_vec[0],0))+'\n'
    return pre

#开始使用flask调整接口
app = Flask(__name__)
sess = model_ini()
@app.route('/upload',methods=['POST'])
def upload():
    f = request.files['file']
    img = f.read()
    pre = Number_recognition(img,sess)
    return pre

@app.route('/')
def index():
    return ''' 
    <!doctype html> 
    <html> 
    <body> 
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'> 
    <input type='submit' value='Upload'> 
    </form> 
    '''

if __name__=='__main__':
    app.debug = True
    app.run(host='0.0.0.0',port=8000)