# Abstract

**Challenges**

- large computation demand to meet the latency and throughput constraints

- variable-length input

  problem->efficient memory manamement and serving optimization

**A transformer serving system**

- computing runtime
- a serving framework

**Three innovative features**

1. An parallel algorithm for batch reduction operations, like **Softmax and LayerNorm**
2. A memory allocation algorithm for **variable-length** input
3. A serving framework with a **batching scheduler** using **DP** achieves the optimal throughput on varaable-length requests

# Introduction

## Compare

Most of current frameworks need a time-consuming **preprocessing** step to tune the computation pattern and memory usage of the operators according to **input dimension**. To solve this problems, they convert variable-length requests into fixed-length requests through **zero paddings**[computational overhead].

None of the existing solutions have investigated serving optimization for variable-length input.

## Solutions

**inference runtime**

- fuse non-GEMM kernels, and implement them efficiently
- conduct light-weight memory usage optimization

**serving framework**

- variable-length-aware batching technique

  combining multiple inference requests into a batch to increase GPU usability

# Design

![image-20201109190021401](..\assets\image-20201109190021401.png)

## Inference Runtime

### Computational Optimizations

#### **Kernel Fusion**

Take PyTorch as an example, whose low efficiency resulting from the following two aspects.

|   batch size and sequence length -> large    | batch size and sequence length -> small |
| :------------------------------------------: | :-------------------------------------: |
|                    20/128                    |                  1/40                   |
|           non-GEMM kernels, 38.2%            |   launch overhead of the CUDA kernels   |
| LayerNorm, Softmax, Add Bias, Transpose, etc |     GPU is idle 80.64% of the time      |

**Advantages**

- reduce the number of memory access
- increase cache locality
- reduce kernel launch overhead

**Scheme**

fuse all the kernels between two GEMM kernels into a single one

![image-20201109192425521](..\assets\image-20201109192425521.png)

non-GEMM kernels can be categorized into two types:

- element-wise operations: activation functions, transpose operations --> parallel
- reduction operations: Softmax, Layer Norm --> focus

**GPU-based Batch-Reduction**

![equation](..\assets\equation.svg)

<center>range[0, 1]    sum=1



LayerNorm lernel

![image-20201109194821533](..\assets\image-20201109194821533.png)

variances:

![image-20201109204506891](..\assets\image-20201109204506891.png)



The first one requires two separate reductions for **x** and **x-E(x)**.

The warpAllReduceSum_2Elem can simultaneously reduce x and x^2 .

![image-20201109205033869](..\assets\image-20201109205033869.png)

### Memory Manager

- The allocation efficiency: the number of times and the amount of the memory is allocated and released
- The memory foorprint: affects the possible size of the model as well as the maximum batch size of requests

Three types of memory are managed by the runtime:

- input tensors
- intermediate tensors
- layer parameters

**Scheme**

Combine the idea of memory cache and graph-topology-aware-space reuse.

- organize memory space is units of the chunks, for example 2MB of memory.
- reuse the same memory space among tensors with no overlapping life cycles.

![image-20201110152206781](..\assets\image-20201110152206781.png)

## Serving Framework

Packaging multiple requests into a relatively larger batch and conducting inference on them together can improve hardware utilization.

How to batch variable-length requests?

![image-20201110164234469](..\assets\image-20201110164234469.png)

![image-20201110164926110](..\assets\image-20201110164926110.png)

When to evoke the batch scheduler?

- hungry strategy: when the runtime is idle
- lazy strategy: timeout value and maximum batch size

# Source Code

./turbo_transformers

​	-->core

​		-->allocator: 内存分配优化的相关cpp文件

​	-->layers: 融合的算子在C++层面的封装，实际调用kernels下的函数

​		-->kernels: 融合算子的CPU实现和GPU实现

​	-->loaders: 用于加载npz模型

​	-->python: 用于测试的python文件和有关model和allocator的python文件

​		-->test: 从huggingface里导入定义好的模型torchmodel，通过from_torch转为turbomodel，进行运行的对比测试。

​		-->turbo_transformers

​			-->layers: 提供了不同模型的from_torch、from_npz、from_pretrained等将普通模型转换为turbo模型的接口

./benchmark: 进行测试对比的py文件及sh脚本

./tools: 创建docker环境和编译测试的shell脚本

./3rd: 第三方工具包

# Install

