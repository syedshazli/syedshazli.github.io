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
Let's start off with making our own shared library with the calculator example from before. Our library will consist of two files, math.c, and its corresponding header file, math.h.
```c file=src/math.c

#include "math.h"

int add(int a, int b){
    return a + b;
}
float mult(int a, int b){
    return a *b;
}
```

```c file=src/math.h

#ifndef MATH_H
#define MATH_H

int add(int a, int b);
float mult(int a, int b);
#endif
```

As you can see, our math library simply provides the functionality to add or multiply two different numbers. Now, let's create a library so we can use the functions provided in math.C.

```bash
gcc -fPIC -c math.c -o math.o
gcc -shared -o libmath.so math.o
```
We use the -fPIC flag to tell the compiler that the generated machine code doesn't depend on any specific memory addresses. The -o flag creates an object file to then be used by the shared library.

To then create the shared library, we add the -shared flag, specifying what we wish to name the shared library, and what object files it will contain. In this case, we'll name our library libmath.so.

To confirm the shared library was created properly:
```bash
$ ls -lh libmath.so
> -rwxrwxrwx 1 root root 15K Dec 15 19:24 libmath.so
```

## Thanks 

Thanks to Professor Andrews of WPI for teaching me what linking and loading is.