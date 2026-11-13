---
title: What Really Matters for New Hardware
author: Syed Shazli
pubDatetime: 2026-11-12T03:07:18Z
slug: dla
featured: true
draft: false
tags:
  - Computer Architecture
  - Hardware
  - GPUs

description: A thought on the state of Deep Learning Accelerators
---
# What Matters
Yes, it is true the cost of iterating on ideas in hardware has become significantly lowered. OpenAI recently released their Jalapeno chip in HotChips, going from ideation to tapeout in about half the time it usually takes. 

So needless to say, with all the rage in deep learning, getting your idea of what hardware the workload should run on will be easier to design.

Even if the design of one chip becomes slightly easier (LLMs still can't reason about Hardware decisions as a whole system), it still becomes extremley hard to tape out a chip that works and is fault tolerant. However, I hypothesize just getting to this step is nowhere near enough to be the big player in serving deep learning (or any) high performance workload.

I hypothesize what matters more is an actual plan for how you will scale up to multi-node communication, from 2 of your chips talking to a million. Distributed systems is a monster of its own and just getting chips to communicate with each other is hard enough. The concept of something like RDMA already amazes me. Your individual chip can only give so much of a performance gain, I want to know how good it is when you scale up. 

The reason why this matters more is n-fold. For one, serving Deep Learning workloads like inference of transformers is a monster of a task. When I did research on MoE at Stanford, it was quite efficient and logical to route one expert to one GPU, but this requires a lot of compute! Once experts needed to communicate, it was clear that if you wanted a good model that takes advantage of MoE, you're gonna need a lot of GPUs to communicate fast and well. Inference serving requires so much processor to processor communication, and if that's slow but your one chip is fast, then good luck.

The same goes for training. Training runs in LLMs span months long, and there is certainly tons of processor communication in that regard. If you fail to do this fast and effectivley, the research lab will simply switch hardware vendors, they can't afford for their training run to break 1 month in because your chip couldn't handle the scale of their operations.

Having a plan for multi-node requires a whole architecture of itself. NVIDIA uses a ring architecture for its NVSWITCH, so that GPUs don't have to spend time bouncing around knowing where they need to go. The switch is the brain of the operation. which was designed with multi-node hardware use cases in mind. It surprised me to see other hardware vendors used flatter architectures, which required travel through many different processors to get to where you need to be. When I hear of all the new chip startups, my first question is not to ask of their single-gpu metrics, but I want to know how much it can scale and how much they've prepared for that.

All's this to say distributed systems is becoming even more of a fascinating topic, and one we should all keep our eye on, not only from the software level, but from the hardware level as well.