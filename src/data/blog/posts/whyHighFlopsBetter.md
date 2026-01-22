---
title: Why More Flops is Better in Parallel Computing
author: Syed Shazli
pubDatetime: 2026-1-16T03:07:18Z
slug: flops
featured: true
draft: true
tags:
  - GPU
  - Parallelism
  - Hardware

description: An explanation of why higher flops is better in high performance computing, even if it may be unintuitive at first.
---

# Abstract
I'm in my second week of my Numerical Methods for Linear and Nonlinear Systems class at WPI. A recurring theme has been to design algorithms that scale, which is very interesting. One thing we discuss a lot is the amount of floating-point operations (FLOPs) of algorithms incur.

Intuitively, more floating point operations should be a bad thing for your algorithms. It causes more steps in your algorithm, which can not only lead to scaling issues when you have vastly large inputs, but this can accumulate floating-point error as well. So why do performance engineers and ML engineers care so much about the FLOP count?

The reason comes down to how parallelism works. A higher 

Modern GPU's have already been designed to allow many FLOPs per second, so the bottleneck becomes memory access. ML workloads mostly have a large amount of data, most of which has to be stored in slower access memory than caches. Thus, the real bottleneck in ML workloads mostly ends up becoming optimizing memory movement.