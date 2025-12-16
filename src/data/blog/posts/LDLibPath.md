---
title: LD_LIBRARY_PATH From First Principles
author: Syed Shazli
pubDatetime: 2025-12-15T07:17:19Z
slug: LDLibPath
featured: true
draft: false
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

We can now write code that references the functions from the shared library.
```c file=src/main.C
#include "math.h"
#include <stdio.h>
int main()
{
    int a = 2;
    int b = 3;
    int result = add(a,b);
    printf("result is %d\n", result);
}
```
## The Problem
Now, let's try to compile and run our program.
```bash
# -I. tells the compiler to find the header file in the current dir at compile time. 
# -L. tells the compiler to find the shared object in the current dir at link time.
# -lmath tells the compiler to link against the library we just created (libmath).
$ gcc main.c -o exe -I. -L. -lmath
$ ./exe

> ./exe: error while loading shared libraries: libmath.so: cannot open shared object file: No such file or directory
```
As you can see, the header file and shared object are all in the same directory as main.c, so there should be no problem! However, we see this error with a shared object.

But we told the compiler to look in the current directory! However, when we run './exe', the dynamic linker searches for libraries in directories such as /usr/lib.

We can confirm our math library was not loaded by running 'ldd exe', which shows all the shared objects used by the program.

```bash
$ ldd exe
    linux-vdso.so.1 (0x00007ffc4a4a2000)
    libmath.so => not found
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f49b2fe5000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f49b321e000)
```

## The Solution

The good news is, the dynamic linker can search for custom paths through LD_LIBRARY_PATH. If we current print LD_LIBRARY_PATH, we get nothing.
```bash
$ echo $LD_LIBRARY_PATH
> 
```
All we have to do is specify the path our shared library is to LD_LIBRARY_PATH!

```bash
$ export LD_LIBRARY_PATH="$PWD:$LD_LIBRARY_PATH"
```
This line sets LD_LIBRARY_PATH to the current directory (where our shared library resides), but can also be set to any directory within your files!

Now, lets see what ldd exe does.
```bash
$ ldd exe
>   linux-vdso.so.1 (0x00007fffb2ff3000)
    libmath.so => /mnt/c/Users/syeda/Downloads/C++ Practice/C--Practice/ldLibPath/tut/libmath.so (0x00007f23bae0e000)
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f23babdc000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f23bae1a000)
```
Nice! The linker finds the shared object file! 

Now, let's run the code:
```bash
$ ./exe
>   result is 5
```

Success!

## Summary

In this post, we got an overview of shared libraries in C, using code from libraries, and using LD_LIBRARY_PATH to allow the dynamic linker to find custom paths to shared libraries.

However, there are some downsides to using LD_LIBRARY_PATH and is not always the best solution. See [this post by Xah Lee](http://xahlee.info/UnixResource_dir/_/ldpath.html) and [this post by DCC](https://www.hpc.dtu.dk/?page_id=1180) for more!

## Thanks 

Thanks to Professor Andrews of WPI for teaching me what linking and loading is!