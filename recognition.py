import caffe
import numpy as np
class recognition():
    def __init__(self):

        self.root='data/'   #根目录
        self.deploy=self.root + 'mnist/deploy.prototxt'    #deploy文件
        self.caffe_model=self.root + 'lenet_iter_18760.caffemodel'   #训练好的 caffemodel
        self.img=self.root+'digit.png'    #随机找的一张待测图片
        self.labels_filename = self.root + 'mnist/test/labels.txt'  #类别名称文件，将数字标签转换回类别名称

        self.net = caffe.Net(self.deploy,self.caffe_model,caffe.TEST)   #加载model和network

#图片预处理设置
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
        self.transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
        self.transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间
        self.transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR

    def output(self):
        im=caffe.io.load_image(self.img)                   #加载图片
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中

        #执行测试
        out = self.net.forward()

        labels = np.loadtxt(self.labels_filename, str, delimiter='\t')   #读取类别名称文件
        prob= self.net.blobs['Softmax1'].data[0].flatten() #取出最后一层（Softmax）属于某个类别的概率值，并打印
        print(prob)
        order=prob.argsort()[-1]  #将概率值排序，取出最大值所在的序号
        print('the class is:',labels[order])   #将该序号转换成对应的类别名称，并打印
        return(labels[order])

if __name__ == '__main__':
    t = recognition()
    t.output()