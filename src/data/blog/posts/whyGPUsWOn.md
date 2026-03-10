---
title: Why Parallel Processors Won
author: Syed Shazli
pubDatetime: 2026-3-10T03:07:18Z
slug: cpi
featured: true
draft: true
tags:
  - GPUs
  - CPUs
  - CPI
  - Parallelism

description: A look at why Parallel Processor won over CPUs through the lens of the CPI equation
---
# Intro
I described in my last blog post about my hardest term yet. Well, I finished the term off well, and wanted to share some things I learned. I want to show why parallel processors (GPUs, TPUs, etc) won: from a CPI perspective. We'll start from 1st principles.

# What is CPI?
CPI, or Clock Cycles per Instruction, is a common metric used to measure the performance of a parallel processor.

A caveat is that the CPI equation can be heavily manipulated to make your processor to seem better than it is. If all instructions tkae 1 clock cycle, but the clock cycle time is very high (usually the amount of time it takes to execute a load instruction), then the CPI is still 1, even if the performance may be poor. As such, the only 'true' metric for processor performance is execution time (of programs, benchmarks, etc).

# Example CPI of a CPU

# Why Parallel Processors won 