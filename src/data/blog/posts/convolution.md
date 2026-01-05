---
title: Convolution in CUDA from First Principles
author: Syed Shazli
pubDatetime: 2025-12-23T03:07:18Z
slug: convolution
featured: true
draft: true
tags:
  - CUDA
  - C++
  - Convolution
  - Machine Learning
  - Performance Analysis

description: A working document of getting convolution in CUDA to work for cuDNN performance.
---

## Intro

In this post, I'll show a working log of how convolution would work in CUDA, from first principles. I was initially motivated to do so after checking out nn.Conv2D module in PyTorch. For most ML programmers, the details of how this is implemented is meant to be abstracted away, but in this post, we try and bring it to light! 