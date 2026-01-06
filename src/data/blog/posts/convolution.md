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
  - Performance Analysis

description: A working document of getting convolution in CUDA to work for cuDNN performance.
---

## Intro

In this post, I'll show a working log of how convolution would work in CUDA, from first principles. I was initially motivated to do so after checking out nn.Conv2D module in PyTorch. For most ML programmers, the details of how this is implemented is meant to be abstracted away, but in this post, we try and bring it to light! To do so, we'll introduce GPU programming, specifically CUDA, and how it's useful for breaking down large 4000x4000 or more images!

Eventually, I hope to get into some optimizations I can do to make our CUDA code run even faster than it alread does.

## What is Convolution?
Convolution is the backbone of Convolutional Neural Networks (CNN's), vital for tasks ranging from facial recognition, self-driving cars, and even stock price prediction!

The point of convolution is to 'combine two functions to produce a third function.' In deep learning, this becomes a useful feature extractor. Convolution can help detect fine grained details that allow an algorithm to make decisions without human assistance.
A example of how convolution works can be seen here: ![Convolution schematic](@/assets/images/convolution_schematic.gif) Figure 1: How Convolution Works. [Source](http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution)

Convolution for images has 3 properties. An input image, its corresponding filter, and the output. In a CNN, the main task is mostly devising an algorithm that can have the optimal filters in order to achieve the task at hand.
## Why do CPUs Struggle With Convolution?

## Where GPUs Come In (And GPU Architecture)

## Naive CUDA Implementation

## Work in Progress: Optimization

## Conclusion