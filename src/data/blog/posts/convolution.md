---
title: Convolution in CUDA from First Principles
author: Syed Shazli
pubDatetime: 2025-12-23T03:07:18Z
slug: convolution
featured: true
draft: false
tags:
  - CUDA
  - C++
  - Convolution
  - Machine Learning
  - Deep Learning
  - Performance Analysis

description: A working document of getting convolution in CUDA to work for cuDNN performance.
---

# [Source Code on Github](github.com/syedshazli/cudaConvolution)
## Intro

In this post, I'll show a working log of how convolution would work in CUDA, from first principles. I was initially motivated to do so after checking out nn.Conv2D module in PyTorch. For most ML programmers, the details of how this is implemented is meant to be abstracted away, but in this post, we try and bring it to light! To do so, we'll introduce GPU programming, specifically CUDA, and how it's useful for breaking down large 4000x4000 or more images!

Eventually, I hope to get into some optimizations I can do to make our CUDA code run even faster than it alread does.

## What is Convolution?
Convolution is the backbone of Convolutional Neural Networks (CNNs), vital for tasks ranging from facial recognition, self-driving cars, and even stock price prediction!

The point of convolution is to 'combine two functions to produce a third function.' In deep learning, this can end up being a useful feature extractor. Convolution can help detect fine grained details that allow an algorithm to make decisions without human assistance (such as the nose of a dog). In Deep Learning models such as ResNet, convolution is repeated multiple times in order to achieve optimal results.

Convolution for images has 3 properties. An input image, its corresponding filter, and the output. In a CNN, the main task is mostly devising an algorithm that can have the optimal filters in order to achieve the task at hand.
A example of how convolution works can be seen here: 
![Convolution schematic](@/assets/images/convolution_schematic.gif) 
Figure 1: How Convolution Works: A simple 2D Convolution Example. [(Source)](http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution)

As you can see, there exists an input matrix, a constant filter that multiplies its element by the current element, and then aggregating the results into one element of the resulting output. The filter then 'slides' over to the next matrix of elements based on something called a 'stride.' (In this case, set to 1). In this blog, we'll assume a stride of 1 when programming, to make things simpler.


## Why do CPUs Struggle With Convolution?
CPUs can do convolution on small images, such as the one shown above, and provide great performance. However, once image dimensions start to increase, the CPU starts to have some trouble. For a 500 x 500 image, the CPU would need to process 250,000 elements sequentially!

The problem is that CPUs are limited by the amount of threads that can be launched to complete a specific task, reaching a couple hundred threads executing concurrently for top of the line server CPUs. The reason for this is because CPUs are not designed to be parallelism machines. Rather, the CPU is optimized to power through any operation as fast as possible. 

Simply put, the amount of threads that need to be launched for a convolution problem is too much for a CPU to handle. Given that multiple convolution passes happens in typical models, this makes for slow performance. Meanwhile, a GPU is designed to solve problems just like this!

## The Power of the GPU
![GPUvsCPU](@/assets/images/cpu-vs-gpu.png)
Figure 2: A simplified diagram displaying CPU vs. GPU architecture. [(Source)](https://tecadmin.net/cpu-vs-gpu-key-differences/)

Graphics Processing Units (GPUs) attempt to curb this problem in a unique way. A GPU is designed to have many more cores than a CPU, enabling a substantial amount of threads to be concurrently running. The tradeoff is that the many cores in a GPU are considerably weaker than that of the CPU. 

GPU vendors such as NVIDIA & AMD release programming models to allow customers to write code to run on the GPU. This leads to much faster training and inference times on Deep Learning and Machine Learning models. We will be using NVIDIA's GPU programming model, known as CUDA, to write code to run on the GPU.

## How the GPU Helps Convolution
There is an inherent form of parallelism that can be exploited by convolution. Each element in the resulting output does not depend on any previous or future elements. So instead of computing each element of the output sequentially, we'll do it at the same time!

Now, let's get into how this can be done in CUDA.
## Naive CUDA Implementation
![threadmapping](@assets/images/threadmapping.png)
The CUDA thread model is shown in Figure 3.

### Vector Addition
I'll first show what a vector addition looks like in CUDA, so you get used to the syntax.
```Cuda
__global__ void add_vectors(double *a, double *b, double *c, int n) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < n) {
c[tid] = a[tid] + b[tid];
}
}
```
As you can see, each thread has its own unique ID. The ID is a culmination of the current thread in the block, the current block in the grid, and the grid dimension, all in the X direction. The __global__ keyword signifies that this function can be run on the GPU and can be called from CPU code. All threads run this same block of code. One array simply stores the addition of elements from two different arrays.
### Convolution
Now, it's time to do convolution.
### Performance
I don't have access to an NVIDIA GPU physically, so I access a NVIDIA GPU by using a remote cluster. In this cluster, I use an A30 GPU and a Intel Xeon Gold 6342 CPU. Needless to say, this is considerably powerful, so results will be much faster than what you may find on your own desktop GPU or laptop.

## Work in Progress: Optimization
I plan on implementing some optimizations to make this run even faster. I'll first try a shared memory approach before moving on to Im2Col.
## Conclusion
In this post, we learned about convolution, the algorithm behind convolutional neural networks (CNNs). We learned why convolution is better suited to run on a GPU, and how to write CUDA code in order to take advantage of the paralellism provided by a GPU to do convolution. Check out the source code [on Github](https://github.com/syedshazli/cudaConvolution) to see the full CUDA kernels as well as my tests.