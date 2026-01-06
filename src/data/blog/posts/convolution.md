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
Convolution is the backbone of Convolutional Neural Networks (CNN's), vital for tasks ranging from facial recognition, self-driving cars, and even stock price prediction!

The point of convolution is to 'combine two functions to produce a third function.' In deep learning, this can end up being a useful feature extractor. Convolution can help detect fine grained details that allow an algorithm to make decisions without human assistance (such as the nose of a dog). In Deep Learning models such as ResNet, convolution is repeated multiple times in order to achieve optimal results.

Convolution for images has 3 properties. An input image, its corresponding filter, and the output. In a CNN, the main task is mostly devising an algorithm that can have the optimal filters in order to achieve the task at hand.
A example of how convolution works can be seen here: ![Convolution schematic](@/assets/images/convolution_schematic.gif) Figure 1: How Convolution Works: A simple 2D Convolution Example. [(Source)](http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution)

As you can see, there exists an input matrix, a constant filter that multiplies its element by the current element, and then aggregating the results into one element of the resulting output. The filter then 'slides' over to the next matrix of elements based on something called a 'stride.' (In this case, set to 1). In this blog, we'll assume a stride of 1 when programming, to make things simpler.


## Why do CPUs Struggle With Convolution?
CPUs can do convolution on small images, such as the one shown above, and provide great performance. However, once image dimensions start to increase, the CPU starts to have some trouble. For a 500 x 500 image, the CPU would need to process 250,000 elements sequentially! Nevermind the fact that image sizes can be much larger for ML models, such as detection of small objects, which can be up to 1500 x 1500 pixels.

The problem is that CPUs are limited by the amount of threads that can be launched to complete a specific task, reaching a couple hundred threads for top of the line server CPUs. The reason for this is because CPUs are not designed to be parallelism machines. Rather, the CPU is more like a bull, optimized to power through any operation as fast as possible. 

Simply put, the amount of threads that need to be launched for a convolution problem is too much for a CPU to handle. Given that multiple convolution passes happens in typical models, this makes for slow performance.

## Where GPUs Come In (And GPU Architecture)
There is an inherent form of paralellism that can be exploited by convolution. Each element in the output does not depend on any previous or future elements.
## Naive CUDA Implementation

## Work in Progress: Optimization

## Conclusion
In this post, we learned about convolution, the algorithm behind convolutional neural networks (CNNs). We learned why convolution is better suited to run on a GPU, and how to write CUDA code in order to take advantage of the paralellism provided by a GPU to do convolution. Check out the source code [on Github](https://github.com/syedshazli/cudaConvolution) to see the full CUDA kernels as well as my tests.