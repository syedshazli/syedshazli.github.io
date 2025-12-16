---
title: The Importance of Defensive Programming
author: Syed Shazli
pubDatetime: 2025-12-16T03:07:18Z
slug: defensive
featured: true
draft: false
tags:
  - C
  - C++
  - Defensive Programming

description: The importance of defensive programming through a quick example.
---

# Intro
I was writing some C code yesterday for [this blog](https://syedshazli.github.io/posts/posts/LDLibPath/) on LD_LIBRARY_PATH, when I encountered a strange bug. My code was the following:
```c file=src/main.C
#include "math.h"
#include <stdio.h>
int main()
{
    int a,b = 6;
    int result = add(a,b); // add function comes from math.h
    printf("result is %d\n", result);
}
```

For some reason, this produced a strange error! I kept running the code, but got garbage values such as 1573957, or -5839328, when I was adding one digit numbers!

I expected the code to initialize 'a' and 'b' as integers, initialized with a value of 6. Needless to say, I spent hours tracking down the issue.

## The Problem

In C, you are allowed to write int a,b = 6, as the compiler will not throw an error. Compared to other languages, like Python, you would get something like this:
```py file = src/main.py
>>> a,b = 6
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: cannot unpack non-iterable int object
```

Given that I was working in C, an error was not thrown to me, and I was confused as to why I was getting wacky outputs. In reality, what was happening was that 'b' was initialized to a value of 6, while 'a' was left uninitialized.

This mean when we went into the 'add' function, we were trying to add a integer of the value 6 with an integer that held no value! I spent at least 15 minutes tracking this bug down, as I was very confused.
## The Benefit of Defensive Programming

If only there was an easier way for errors like this to be seen by me! And the truth is, there is! This is the premise of what some call 'defensive programming.' The idea is to validate inputs and outputs and do rigorous checking before and after doing important work with your data.

Defensive programming will allow you to clearly view errors in your code that otherwise wouldn't have been thrown by the compiler, and can save you loads of time. For example, it is (mostly) common practice to not check if the return type of 'malloc' returned a nullptr. But if 'malloc' really did return a null pointer for whatever reason, you would be left scratching your head for some time!

Of course, defensive programming is more than just rigorous checking in the code itself. You can also implement workflows such as 'treat all warnings as errors' or enabling warnings to show up in compilation. 

In my case, I compiled as 
```bash
gcc main.c -o exe -L. -lmath
```
Which does not throw any warnings related to 'a' being uninitialized. However, I don't even need to rewrite any code defensivley to find the problem. I simply add a compilation flag for gcc named 'Wall,' which enables a large amount of compiler warnings to show up in my code should my code bring up these errors.

```bash
gcc main.c -o exe -L. -Wall -lmath
>   main.c: In function ‘main’:
    main.c:6:7: warning: ‘a’ is used uninitialized [-Wuninitialized]
    6 |     if(a){
      |       ^
```

Bug found!


## Conclusion

In this post, I wrote about a silly bug I wrote in my code, and how defensive programming could have made the bug a lot easier to fix! I hope you learned something about defensive programming. I sure did, and will work to compile my C/C++ code with -Wall in the future, treating all warnings as errors!