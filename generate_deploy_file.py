from caffe import layers as L,params as P,to_proto
root='data/'
deploy=root+'mnist/deploy.prototxt'    #文件保存路径

def create_deploy():
    #少了第一层，data层
    conv1 = L.Convolution(bottom = "data", kernel_size=5, stride=1, num_output=20, pad=4, weight_filler=dict(type='xavier'))
    # 激活函数层
    relu1 = L.ReLU(conv1, in_place=True)
    # 卷积层
    conv2 = L.Convolution(relu1, kernel_size=5, stride=1, num_output=50, pad=4, weight_filler=dict(type='xavier'))
    # 激活函数层
    relu2 = L.ReLU(conv2, in_place=True)
    # 池化层
    pool1 = L.Pooling(relu2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # 第二层：卷积层
    conv3 = L.Convolution(pool1, kernel_size=5, stride=1, num_output=20, pad=4, weight_filler=dict(type='xavier'))
    # 激活函数层
    relu3 = L.ReLU(conv3, in_place=True)
    # 卷积层
    conv4 = L.Convolution(relu3, kernel_size=5, stride=1, num_output=50, pad=4, weight_filler=dict(type='xavier'))
    # 激活函数层
    relu4 = L.ReLU(conv4, in_place=True)
    # 全连接层
    fc3 = L.InnerProduct(relu4, num_output=500, weight_filler=dict(type='xavier'))
    # 激活函数层
    relu3 = L.ReLU(fc3, in_place=True)
    # 全连接层
    fc4 = L.InnerProduct(relu3, num_output=10, weight_filler=dict(type='xavier'))
    #最后没有accuracy层，但有一个Softmax层
    prob=L.Softmax(fc4)
    return to_proto(prob)
def write_deploy():
    with open(deploy, 'w') as f:
        f.write('name:"Lenet"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:28\n')
        f.write('input_dim:28\n')
        f.write(str(create_deploy()))
if __name__ == '__main__':
    write_deploy()
