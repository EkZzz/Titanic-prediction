```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
```


```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(len(train_data), len(test_data))
train_data.head()
```

    891 418
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



# 打乱训练集


```python
train_data = train_data.sample(frac=1)
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>882</th>
      <td>883</td>
      <td>0</td>
      <td>3</td>
      <td>Dahlberg, Miss. Gerda Ulrika</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>7552</td>
      <td>10.5167</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>568</th>
      <td>569</td>
      <td>0</td>
      <td>3</td>
      <td>Doharr, Mr. Tannous</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2686</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36</td>
      <td>0</td>
      <td>1</td>
      <td>Holverson, Mr. Alexander Oskar</td>
      <td>male</td>
      <td>42.0</td>
      <td>1</td>
      <td>0</td>
      <td>113789</td>
      <td>52.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>211</th>
      <td>212</td>
      <td>1</td>
      <td>2</td>
      <td>Cameron, Miss. Clear Annie</td>
      <td>female</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>F.C.C. 13528</td>
      <td>21.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>346</th>
      <td>347</td>
      <td>1</td>
      <td>2</td>
      <td>Smith, Miss. Marion Elsie</td>
      <td>female</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>31418</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



# 总体数据缺省
## 合并起来对缺省数据填充或删除

<table><thead><tr><th>变量</th><th>定义</th><th>键值</th></tr></thead><tbody><tr><td>survival</td><td>存活</td><td>0=No,1=Yes</td></tr><tr><td>pclass</td><td>票的类别</td><td>1=1st,2=2nd,3=3rd</td></tr><tr><td>sex</td><td>性别</td><td></td></tr><tr><td>Age</td><td>年龄</td><td></td></tr><tr><td>sibsp</td><td>在船上有几个兄弟/配偶</td><td></td></tr><tr><td>parch</td><td>在船上有几个双亲/孩子</td><td></td></tr><tr><td>ticket</td><td>票编号</td><td></td></tr><tr><td>fare</td><td>乘客票价</td><td></td></tr><tr><td>cabin</td><td>客舱号码</td><td></td></tr><tr><td>embarked</td><td>登船港口</td><td>C = Cherbourg, Q = Queenstown, S = Southampton</td></tr></tbody></table>


```python
all_data=pd.concat([train_data,test_data],ignore_index=True)
all_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  1309 non-null   int64  
     1   Survived     891 non-null    float64
     2   Pclass       1309 non-null   int64  
     3   Name         1309 non-null   object 
     4   Sex          1309 non-null   object 
     5   Age          1046 non-null   float64
     6   SibSp        1309 non-null   int64  
     7   Parch        1309 non-null   int64  
     8   Ticket       1309 non-null   object 
     9   Fare         1308 non-null   float64
     10  Cabin        295 non-null    object 
     11  Embarked     1307 non-null   object 
    dtypes: float64(3), int64(4), object(5)
    memory usage: 122.8+ KB
    

## 处理数据


```python
# 对缺省的年龄进行平均数填充
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].mean())
# 对缺省的登船港口进行众数填充
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode())
# 性别转换为1和0
all_data['Sex']=[1 if x=='male' else 0 for x in all_data.Sex]
# 将SibSp,Parch 用有无代替数量以减少离散
all_data['SibSp']=[0 if x==0 else 1 for x in all_data.Sex]
all_data['Parch']=[0 if x==0 else 1 for x in all_data.Sex]
# 丢失票价用同等票类型的平均数进行填充
all_data['Fare'] = all_data['Fare'].fillna(all_data.pivot_table(index='Pclass', values='Fare', aggfunc=np.mean).loc[all_data[all_data['Fare'].isnull()]['Pclass'],'Fare'])
# 将上船码头与票价以独热编码鵆
all_data['p1']=np.array(all_data['Pclass']==1).astype(np.int32)
all_data['p2']=np.array(all_data['Pclass']==2).astype(np.int32)
all_data['p3']=np.array(all_data['Pclass']==3).astype(np.int32)
all_data['e1']=np.array(all_data['Embarked']=='S').astype(np.int32)
all_data['e2']=np.array(all_data['Embarked']=='C').astype(np.int32)
all_data['e3']=np.array(all_data['Embarked']=='Q').astype(np.int32)
#对票价和年龄归一化
all_data['Age'] = (all_data['Age']-all_data['Age'].min())/(all_data['Age'].max()-all_data['Age'].min())
all_data['Fare'] = (all_data['Fare']-all_data['Fare'].min())/(all_data['Fare'].max()-all_data['Fare'].min())
```

## 数据选取


```python
all_data_selected = all_data[['p1', 'p2', 'p3', 'e1', 'e2', 'e3', 'Sex', 'Fare', 'Age']].values#, 'SibSp', 'Parch'
train_data = all_data_selected[:len(train_data)]
test_data = all_data_selected[len(train_data):]
train_label1 = all_data[['Survived']].values
train_lable = train_label1[:len(train_data)].reshape(-1,1)

train_data[0:5,:]
```




    array([[0.        , 0.        , 1.        , 1.        , 0.        ,
            0.        , 0.        , 0.02052723, 0.27345609],
           [0.        , 0.        , 1.        , 0.        , 1.        ,
            0.        , 1.        , 0.01411046, 0.3721801 ],
           [1.        , 0.        , 0.        , 1.        , 0.        ,
            0.        , 1.        , 0.10149724, 0.52398848],
           [0.        , 1.        , 0.        , 1.        , 0.        ,
            0.        , 0.        , 0.04098927, 0.43630214],
           [0.        , 1.        , 0.        , 1.        , 0.        ,
            0.        , 0.        , 0.02537431, 0.49893524]])




```python
train_x = tf.cast(train_data, tf.float32)
train_y = tf.cast(train_lable, tf.int16)
test_x = tf.cast(test_data, tf.float32)
print(train_x.shape, train_y.shape)
```

    (891, 9) (891, 1)
    

## 模型配置与训练


```python
model=tf.keras.Sequential()
try:
    model = tf.keras.models.load_model("Titanic.h5")
    model.summary()
except:
    model.add(tf.keras.layers.Dense(16,activation="relu",kernel_regularizer = tf.keras.regularizers.l2()))#input_shape=(1,9)添加隐含层，隐含层是全连接层，128个结点,激活函数使用relu函数
    model.add(tf.keras.layers.Dense(8,activation="relu"))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))#添加输出层，输出层使全连接层，激活函数是sigmoid函数
    
    #配置训练方法
    #优化器使用adam,损失函数使用二值交叉熵损失函数，准确率使用稀疏分类准确率函二值
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.07),
                loss='binary_crossentropy',
                metrics=['binary_accuracy'])
    
    # #训练模型
    # #使用训练集中的数据训练，从中划分20%作为测试数据,用在每轮训练后评估模型的性能，每个小批量使用32条数据，训练50轮
    history = model.fit(train_x,train_y,
                batch_size=32,
                epochs=60,
                validation_split=0.2)
    model.summary()
    model.save("Titanic.h5", overwrite=True, save_format=None)
```

    Epoch 1/60
    23/23 [==============================] - 2s 46ms/step - loss: 0.6572 - binary_accuracy: 0.6516 - val_loss: 0.5127 - val_binary_accuracy: 0.8045
    Epoch 2/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.5755 - binary_accuracy: 0.7653 - val_loss: 0.5096 - val_binary_accuracy: 0.8101
    Epoch 3/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.5329 - binary_accuracy: 0.7950 - val_loss: 0.4760 - val_binary_accuracy: 0.8101
    Epoch 4/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4905 - binary_accuracy: 0.7873 - val_loss: 0.5239 - val_binary_accuracy: 0.8045
    Epoch 5/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4977 - binary_accuracy: 0.8053 - val_loss: 0.4804 - val_binary_accuracy: 0.8101
    Epoch 6/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4874 - binary_accuracy: 0.8013 - val_loss: 0.4880 - val_binary_accuracy: 0.8045
    Epoch 7/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4855 - binary_accuracy: 0.8014 - val_loss: 0.4698 - val_binary_accuracy: 0.7933
    Epoch 8/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4985 - binary_accuracy: 0.8015 - val_loss: 0.4595 - val_binary_accuracy: 0.7989
    Epoch 9/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4720 - binary_accuracy: 0.7949 - val_loss: 0.4923 - val_binary_accuracy: 0.7989
    Epoch 10/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.5137 - binary_accuracy: 0.7955 - val_loss: 0.4798 - val_binary_accuracy: 0.8045
    Epoch 11/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.5243 - binary_accuracy: 0.7728 - val_loss: 0.4830 - val_binary_accuracy: 0.7877
    Epoch 12/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4612 - binary_accuracy: 0.8014 - val_loss: 0.4804 - val_binary_accuracy: 0.8101
    Epoch 13/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4632 - binary_accuracy: 0.8163 - val_loss: 0.5068 - val_binary_accuracy: 0.7765
    Epoch 14/60
    23/23 [==============================] - 0s 10ms/step - loss: 0.4824 - binary_accuracy: 0.8045 - val_loss: 0.4965 - val_binary_accuracy: 0.7877
    Epoch 15/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4676 - binary_accuracy: 0.7947 - val_loss: 0.4939 - val_binary_accuracy: 0.8045
    Epoch 16/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4991 - binary_accuracy: 0.7984 - val_loss: 0.4746 - val_binary_accuracy: 0.8045
    Epoch 17/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4653 - binary_accuracy: 0.8099 - val_loss: 0.5367 - val_binary_accuracy: 0.7765
    Epoch 18/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.5560 - binary_accuracy: 0.7619 - val_loss: 0.4881 - val_binary_accuracy: 0.7933
    Epoch 19/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4831 - binary_accuracy: 0.8072 - val_loss: 0.4921 - val_binary_accuracy: 0.7877
    Epoch 20/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4963 - binary_accuracy: 0.7795 - val_loss: 0.4626 - val_binary_accuracy: 0.7989
    Epoch 21/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4710 - binary_accuracy: 0.7985 - val_loss: 0.4645 - val_binary_accuracy: 0.7989
    Epoch 22/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.5250 - binary_accuracy: 0.7593 - val_loss: 0.5046 - val_binary_accuracy: 0.7933
    Epoch 23/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4925 - binary_accuracy: 0.7697 - val_loss: 0.5046 - val_binary_accuracy: 0.8045
    Epoch 24/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.5215 - binary_accuracy: 0.8060 - val_loss: 0.4720 - val_binary_accuracy: 0.7989
    Epoch 25/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.5046 - binary_accuracy: 0.7788 - val_loss: 0.4782 - val_binary_accuracy: 0.8156
    Epoch 26/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4875 - binary_accuracy: 0.7981 - val_loss: 0.4633 - val_binary_accuracy: 0.8101
    Epoch 27/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4506 - binary_accuracy: 0.8066 - val_loss: 0.4945 - val_binary_accuracy: 0.8045
    Epoch 28/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4954 - binary_accuracy: 0.7996 - val_loss: 0.4632 - val_binary_accuracy: 0.8045
    Epoch 29/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.5015 - binary_accuracy: 0.7957 - val_loss: 0.4626 - val_binary_accuracy: 0.8156
    Epoch 30/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4371 - binary_accuracy: 0.8214 - val_loss: 0.4912 - val_binary_accuracy: 0.8045
    Epoch 31/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4997 - binary_accuracy: 0.8070 - val_loss: 0.4712 - val_binary_accuracy: 0.7989
    Epoch 32/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4823 - binary_accuracy: 0.7830 - val_loss: 0.4536 - val_binary_accuracy: 0.8101
    Epoch 33/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4635 - binary_accuracy: 0.8050 - val_loss: 0.4739 - val_binary_accuracy: 0.8212
    Epoch 34/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4660 - binary_accuracy: 0.8076 - val_loss: 0.4661 - val_binary_accuracy: 0.7989
    Epoch 35/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4518 - binary_accuracy: 0.7977 - val_loss: 0.4990 - val_binary_accuracy: 0.8101
    Epoch 36/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.5017 - binary_accuracy: 0.8100 - val_loss: 0.4763 - val_binary_accuracy: 0.7877
    Epoch 37/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4720 - binary_accuracy: 0.8239 - val_loss: 0.4682 - val_binary_accuracy: 0.8101
    Epoch 38/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4611 - binary_accuracy: 0.8247 - val_loss: 0.4740 - val_binary_accuracy: 0.8045
    Epoch 39/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4819 - binary_accuracy: 0.7977 - val_loss: 0.4650 - val_binary_accuracy: 0.7989
    Epoch 40/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4580 - binary_accuracy: 0.8010 - val_loss: 0.4781 - val_binary_accuracy: 0.7877
    Epoch 41/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.5039 - binary_accuracy: 0.7742 - val_loss: 0.4800 - val_binary_accuracy: 0.8045
    Epoch 42/60
    23/23 [==============================] - 0s 7ms/step - loss: 0.4637 - binary_accuracy: 0.8137 - val_loss: 0.4608 - val_binary_accuracy: 0.8045
    Epoch 43/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4755 - binary_accuracy: 0.7879 - val_loss: 0.4475 - val_binary_accuracy: 0.8045
    Epoch 44/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4979 - binary_accuracy: 0.7690 - val_loss: 0.4619 - val_binary_accuracy: 0.8101
    Epoch 45/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4610 - binary_accuracy: 0.8041 - val_loss: 0.4763 - val_binary_accuracy: 0.8156
    Epoch 46/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4965 - binary_accuracy: 0.8080 - val_loss: 0.5072 - val_binary_accuracy: 0.7765
    Epoch 47/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.5597 - binary_accuracy: 0.7517 - val_loss: 0.4739 - val_binary_accuracy: 0.7989
    Epoch 48/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4957 - binary_accuracy: 0.7976 - val_loss: 0.4744 - val_binary_accuracy: 0.7933
    Epoch 49/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4632 - binary_accuracy: 0.8113 - val_loss: 0.4645 - val_binary_accuracy: 0.8045
    Epoch 50/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4358 - binary_accuracy: 0.8259 - val_loss: 0.5332 - val_binary_accuracy: 0.7877
    Epoch 51/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.5262 - binary_accuracy: 0.7589 - val_loss: 0.4734 - val_binary_accuracy: 0.7821
    Epoch 52/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4488 - binary_accuracy: 0.8031 - val_loss: 0.4984 - val_binary_accuracy: 0.8045
    Epoch 53/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4955 - binary_accuracy: 0.7854 - val_loss: 0.5163 - val_binary_accuracy: 0.7877
    Epoch 54/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.5196 - binary_accuracy: 0.7652 - val_loss: 0.4999 - val_binary_accuracy: 0.7877
    Epoch 55/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.5786 - binary_accuracy: 0.7633 - val_loss: 0.5107 - val_binary_accuracy: 0.7877
    Epoch 56/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.5229 - binary_accuracy: 0.7721 - val_loss: 0.4679 - val_binary_accuracy: 0.8156
    Epoch 57/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4780 - binary_accuracy: 0.8111 - val_loss: 0.5396 - val_binary_accuracy: 0.7709
    Epoch 58/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4708 - binary_accuracy: 0.7908 - val_loss: 0.4791 - val_binary_accuracy: 0.8101
    Epoch 59/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4723 - binary_accuracy: 0.7982 - val_loss: 0.4669 - val_binary_accuracy: 0.7933
    Epoch 60/60
    23/23 [==============================] - 0s 6ms/step - loss: 0.4838 - binary_accuracy: 0.7895 - val_loss: 0.4524 - val_binary_accuracy: 0.8101
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 16)                160       
    _________________________________________________________________
    dense_1 (Dense)              (None, 8)                 136       
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 9         
    =================================================================
    Total params: 305
    Trainable params: 305
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.evaluate(train_x,train_y)
```

    28/28 [==============================] - 0s 3ms/step - loss: 0.4559 - binary_accuracy: 0.8092
    




    [0.45588964223861694, 0.8092031478881836]



## 可视化


```python
#accuracy的历史
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'], loc='upper left')
plt.ylim((0, 1))
plt.show()
 
#loss的历史
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc='upper left')
plt.ylim((0, 1))
plt.show()
```


    
![png](titanic_files/titanic_16_0.png)
    



    
![png](titanic_files/titanic_16_1.png)
    


## 测试结果评价


```python
test_y = pd.read_csv('gender_submission.csv')
test_y = np.array(test_y['Survived'])
model.evaluate(test_x,test_y)
```

    14/14 [==============================] - 0s 3ms/step - loss: 0.2923 - binary_accuracy: 0.8900
    




    [0.2923037111759186, 0.8899521827697754]



## 保存数据
* 保存预测生存状况
* 保存预测生存概率


```python
pred = model.predict(test_x)
result = pd.DataFrame({'PassengerId':np.arange(892,892+418), 'Survived':tf.where(pred>0.5,1,0).numpy().reshape(-1)})
result.to_csv("titanic_survived_predictions.csv", index=False)
```


```python
result = pd.DataFrame({'PassengerId':np.arange(892,892+418), 'Survived':pred.reshape(-1)})
result.to_csv("titanic_psurvived_predictions.csv", index=False)
```
