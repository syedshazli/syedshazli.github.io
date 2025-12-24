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

I was inspired by [this post](https://siboehm.com/articles/22/CUDA-MMM) by Simon Boehm on implementing Matrix Multiplication in CUDA from first principles, eventually optimizing enough to get CuBLAS performance.