#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from keras.datasets import cifar10


# In[2]:


(X_train,y_train),(X_test,y_test) = cifar10.load_data()


# In[3]:


X_train.shape


# In[ ]:





# In[4]:


from keras.datasets import mnist


# In[5]:


(X_train,y_train),(X_test,y_test) = mnist.load_data()


# In[6]:


X_train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


# 加载函数库
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
#from keras.datasets import mnist, cifar10
from keras.datasets import cifar10
from keras.utils import np_utils
import keras
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils.vis_utils import plot_model


# In[16]:


# 初始化模型参数
# 随机种子
np.random.seed(666)
# 训练轮数
NB_EPOCH = 5
# batch size
BATCH_SIZE = 256
# 四层神经元数量总数
N_ADDITION = 1024
#verbose：日志显示
#verbose = 0 为不在标准输出流输出日志信息
#verbose = 1 为输出进度条记录
#verbose = 2 为每个epoch输出一行记录
VERBOSE = 1
# 分类标签种类数量
NB_CLASSES = 10

#OPTIMIZER = SGD()
# 隐藏层神经元数量
# 第一层
N_HIDDEN = 128
#validation_split用于在没有提供验证集的时候
#按一定比例从训练集中取出一部分作为验证集
VALIDATION_SPLIT = 0.2


# In[25]:


# 准备数据
(X_train,y_train),(X_test,y_test) = cifar10.load_data()
# convolution不需要做数据结构的修改
# 还是32*32， 不用转化为一维数组
# 数据类型
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
# 数据预处理 
# 图片shape
input_shape = (32,32,3)
print(X_train.shape)
X_train /= 255
X_test  /= 255
# 分类标签处理:转化为二值化序列
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test  = np_utils.to_categorical(y_test,NB_CLASSES)


# In[26]:


# 构建卷积模型
model = Sequential()
model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))


# In[27]:


# 初始化回调函数
CSV_log = keras.callbacks.CSVLogger(filename='csv/cnn_CIFAR10.log', separator=',', append=False)
# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# 训练模型
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH
              ,verbose=VERBOSE
          , validation_split=VALIDATION_SPLIT
          , callbacks=[CSV_log])
   


# In[28]:


# 模型评分
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("test loss     : ", score[0])
print("test accuracy : ", score[1])
# summary()用于显示模型详细信息
model.summary()
# plot_model函数用于绘制神经网络模型结构
# 输入所需要绘制模型的model
# 输出图片为model.png,并展示各网络层的大小
plot_model(model, to_file='model_convolution_CIFAR10.png',show_shapes=True)


# In[29]:


# 对日志进行抽取并显示
from pandas import DataFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[30]:


# 读取各个不同的日志文件
path = 'csv\cnn_CIFAR10.log'
data = pd.read_csv(path, sep=',')
# 配置排版
plt.subplot()
#绘制曲线
sns.lineplot(data=data[['accuracy','val_accuracy']])
# 设置标题
plt.title('model CNN_CIFAR10:'
          '\nbest acc is {0:.4}'
          '\nbest val_acc is {1:.4}'.format(data[['accuracy']].max().values[0], data[['val_accuracy']].max().values[0])
         )
    

#plt.figure(figsize=(16, 12))
#plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




