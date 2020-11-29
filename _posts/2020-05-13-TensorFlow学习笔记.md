---
title:  "初学TensorFlow的一些记录"
date:   2020-5-14 15:43:58
tags: TensorFlow
---
# 函数篇

- ##### tf.argmax(input, axis)

```python
test = np.array([      #shape(test) = 4 × 3
[1, 2, 3],
 [2, 3, 4], 
 [5, 4, 3], 
 [8, 7, 2]])
np.argmax(test, 0)　　　#输出：array([3, 3, 1]     
#第0维度对应4，即在第1维固定的条件下第0维的4个元素中选择最大的元素，第1维有3个取值，对应3个索引
np.argmax(test, 1)　　　#输出：array([2, 2, 0, 0]
#第1维度对应3，即在第0维固定的条件下第1维的3个元素中选择最大的元素，第0维有4个取值，对应4个索引
```

- ##### tf.reduce_max(input_tensor, axis=None, keepdims=False, name=None)

```python
a=np.array([[1, 2],
            [5, 3],
            [2, 6]])
b = tf.Variable(a)
tf.reduce_max(b, axis=1, keepdims=False)
#[2 5 6]
tf.reduce_max(b, axis=1, keepdims=True)
#[[2]
#[5]
#[6]]
tf.reduce_max(b, axis=0, keepdims=True)
#[[5 6]]
```

- ##### tf.slice(input, begin, size, name=None)

```python
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
# shape=[3, 2, 3]
tf.slice(t, [1, 0, 0], [1, 1, 3])
[[[3, 3, 3]]]
```

- ##### tf.concat(axis, tensor_list)

```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat(0, [t3, t4])) ==> [4, 3]
tf.shape(tf.concat(1, [t3, t4])) ==> [2, 6]
```

# 实验室服务器使用篇

- ##### 设置使用固定的GPU：

`CUDA_VISIBLE——DEVICES={GPU标号}  python main.py`

# 错误处理篇

- ##### TypeError: pred must not be a Python bool

解决办法：`is_training = tf.cast(True, tf.bool)`