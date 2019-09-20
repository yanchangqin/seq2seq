import tensorflow as tf
import numpy as np
from getcode import Imagedate
import PIL.Image as image
import PIL.ImageDraw as draw
import PIL.ImageFont as Font
import matplotlib.pyplot as plt

batch =100
font = Font.truetype('1.TTF',size=10)

class Encoder_net:
    def __init__(self):
        #NHWC(100,60,120,3)-->NWHC(100,120,60,3)-->NW*HC(100*120,60*3)
        self.w1 = tf.Variable(tf.truncated_normal(shape=[60*3,128]))
        self.b1 = tf.Variable(tf.zeros([128]))
    def forward(self,x):
        with tf.name_scope('Encoder_net') as scope:
            #NHWC(100,60,120,3)-->NWHC(100,120,60,3)
            y = tf.transpose(x,[0,2,1,3])
            #NW*HC--->NV
            y = tf.reshape(y,[batch*120,60*3])
            y1 = tf.nn.relu(tf.matmul(y,self.w1)+self.b1)
            #y1-->[100,120,128]
            y1 =tf.reshape(y1,[batch,120,128])
            # print(y1.shape)

            cell = tf.nn.rnn_cell.BasicLSTMCell(128,name = 'encoder_cell')
            init_state = cell.zero_state(batch,dtype=tf.float32)
            en_output,en_finalstate = tf.nn.dynamic_rnn(cell,y1,initial_state =init_state )
            y = tf.transpose(en_output,[1,0,2])[-1]
        return y#[100,128]

class Decoder_net:
    def __init__(self):
        self.w = tf.Variable(tf.truncated_normal(shape=[128,10]))
        self.b = tf.Variable(tf.zeros([10]))
    def forward(self,x):
        #NV(100,128)--->NSV(100,1,128)
        y = tf.expand_dims(x,axis =1)
        #(100,4,128)
        y = tf.tile(y,[1,4,1])
        cell = tf.nn.rnn_cell.LSTMCell(128, name='decoder_cell')
        init_state = cell.zero_state(batch, dtype=tf.float32)
        de_output, de_final_state = tf.nn.dynamic_rnn(cell, y, initial_state=init_state)
        y = tf.reshape(de_output,[batch*4,128])
        y1 = tf.matmul(y,self.w)+self.b
        self.out = tf.reshape(y1,[batch,4,10])
        return self.out

class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[batch,60,120,3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[batch,4,10])
        self.encoder = Encoder_net()
        self.decoder = Decoder_net()
    def forward(self):
        y = self.encoder.forward(self.x)
        self.output = self.decoder.forward(y)
    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.output))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    igd = Imagedate()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)
    accs = []
    max_acc=0
    with tf.Session() as sess:
        sess.run(init)
        for i in range(50000):
            xs,ys = igd.get_code(batch)
            error,_ = sess.run([net.loss,net.optimizer],feed_dict={net.x:xs,net.y:ys})
            if i %100 == 0:
                xss,yss =igd.get_code(batch)
                _error,out = sess.run([net.loss,net.output],feed_dict={net.x:xss,net.y:yss})
                output = np.argmax(out[0],axis=1)
                label = np.argmax(yss[0],axis=1)
                acc = np.mean(np.array(np.argmax(out,axis=2) == np.argmax(yss,axis=2),dtype=np.float32))
                accs.append(acc)
                print(accs)
                print('最大精度：',max(accs))
                # print('损失：',error)
                # print('标签：',label)
                # print('输出：',output)
                print('精度：',acc)
                # if acc>=max(accs):
                if acc>=max_acc:
                    max_acc=acc
                    saver.save(sess,"./params/ckpt")
                    print('已存')
                else:
                    print('pass')
                # x = (np.array(xss)[0]+0.5)*255
                # img = image.fromarray(np.uint8(x,mode='RGB'))
                # img1 = draw.ImageDraw(img)
                # img1.text((5,5),text=str(output),fill=(255,255,255),font=font)
                # plt.imshow(img)
                # plt.pause(0.1)








