---
title: Exploring LD_LIBRARY_PATH
author: Syed Shazli
pubDatetime: 2025-12-15T05:17:19Z
slug: LDLibPath
featured: true
draft: true
tags:
  - C
  - Computer Systems
  - Linux
  - Tutorial

description: This is a post that goes over creating your own shared libraries, and linking against them in C/C++ using LD_LIBRARY_PATH.
---

# Intro

I wanted to start my first technical blog post relating to something I struggled with a lot over the summer: LD_LIBRARY_PATH. I would constantly run into issues where my code wasn't building even though I saw the libraries it linked to clearly existed.

This blog post will go over:
1. What are shared libraries and how to create one.
2. Execution issues with code referenced from shared libraries.
3. Solving linking issues with LD_LIBRARY_PATH

## What Are Shared Libraries?
Shared libraries are a collection of one or more files packaged together in order to be referenced by programs at run-time. Programs using shared libraries only references the code it needs from that library. Libraries end up being useful when there's a collection of functions that a programmer can use for a specific task. 

For example, one might make a calculator library that stores the various functions of a calculator (adding, subtracting, etc). The programmer no longer needs to implement these functions, and can call upon the function to commit an operation of their desire.

In Linux, shared libraries are seen as .so (shared object) files. In Windows, these are seen as .dll (dynamically linked library) files.

Static libraries also exist, and are oftentimes uses by programmers. However, the entire content of the library ends up becoming embedded within the executable, mkaing the executable size much larger.

### Example
Let's start off with making our own shared library with the calculator example from before. Our library will consist of one file, math.c.
```c
// Math.c
#include "Math.h"

int add(int a, int b){
    return a + b;
}
float mult(int a, int b){
    return a *b;
}
```

```c
// Math.h
#ifndef MYMATH_H
#define MYMATH_H

int add(int a, int b);
float mult(int a, int b);
#endif
```



## Thanks 

Thanks to Professor Andrews of WPI for teaching me what linking and loading is.