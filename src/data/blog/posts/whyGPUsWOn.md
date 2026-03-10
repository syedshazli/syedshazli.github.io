---
title: Why Parallel Processors Won
author: Syed Shazli
pubDatetime: 2026-3-10T03:07:18Z
slug: cpi
featured: true
draft: false
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

CPI is calculated as the sum of the frequency of an instruction (a ratio) * the latency of each instruction. A high CPI is generally a bad thing, implying your processors is taking up a lot of clock cycles to execute an instruction. Vice versa, a low CPI implies your processor takes a small amount of time on average to execute an instruction, which is a good thing.

As you may notice, the frequency of each instruction is heavily dependent on which program you run. For this reason we use benchmarks, which are software programs meant to simulate the workload for specific users. Benchmarks exist for gaming, AI, etc. 

A caveat is that the CPI equation can be heavily manipulated to make your processor to seem better than it is. If all instructions take 1 clock cycle, but the clock cycle time is very high (usually the amount of time it takes to execute a load instruction), then the CPI is still 1, even if the performance may be poor. As such, the only 'true' metric for processor performance is execution time (of programs, benchmarks, etc).

# Example CPI of a CPU
Let's say we have a program that has 30% of instructions being ALU instructions, 50% are load/store instructions, and 20% are branch instructions. Our processor takes 3 clock cycles for load, 1 for branch, and 1 for ALU. The CPI would be  1 * 0.3 + 3 * 0.5 + 1 * 0.2, which is 0.3 + 1.5 + 0.2, so the CPI is 2.

# Why Parallel Processors Won 
The execution time of a program is defined as the number of instructions in a programs multiplied by the clock rate multiplied by the CPI. CPUs are known as blazing fast machines. They must be in order to execute a broad range of instructions fast. After all, the goal of a CPU is to execute instructions as fast as possible. As a result, in the CPI equation, the latency of each instruction for CPUs is minimized as much as possible.

GPUs and parallel processors take a different approach. Instead of attempting to execute instructions as fast as possible, why don't we increase the instruction execution time but have multiple results arrive from that instruction, not just 1. In order to do this, hardware support needs to be enabled to complete work independently at the same time, introducing more 'cores' to a processor. This is known as Single-Instruction, Multiple Data (SIMD). CPUs can support SIMD instructions, but the hardware isn't optimized to do so, the hardware is optimized to execute instructions fast. Parallel processors add hundreds of cores with special units inside each one meant to execute SIMD instructions.

As a result, the clock cycle per instruction might be high, but the actual execution time goes down due to so many results arriving from one instruction!