---
title: Simulators Are All You Need
author: Syed Shazli
pubDatetime: 2026-10-07T07:17:19Z
slug: Simulators
featured: true
draft: false
tags:
  - Computer Systems

description: Why Simulators Are the Future
---


# Simulators Are All You Need
For years, people have tried to determine how we can get optimal value out of something we don't physically have yet. This can be achieved through software simulation.

However, simulation hasn't always been in software. Before the 1900's, generals would simulate war by playing war-based games, with physical scale models of what resources they would need. John von Neumann used simulators while he was at the manhattan project to develop random sampling algorithms to simulate neutron diffusion.

The first time I've heard of simulators and their use case in a formal classroom setting, I was blown away. The fact that we have software based simulators that not only simulate our desired hardware specifications, but simulate down to the cycle, was fascinating. This really made me realise why and how the opportunities software provides is endless.

Jensen Huang was asked on the Dwarkesh Podcast on why they don't just build different families of the same architecture, optimized for different user needs. His answer? "We simulated it in software and found it wasn't more efficient than the solutions we have today." (paraphrased)

Simulation is (likely) the backbone of all hardware & computer architecture research in general. How can you gain funding to make a big architecture change if you have no proof it works? You make a software based simulation of it, and that's the next best thing compared to taping out the chip and iterating on the effects after the fact.

Fast-forward to nowadays, simulation in accelerated computing has gone far beyond simulating components of a computer. Frontier companies will also simulate how their deep learning models will run before making a training run that may run for months!

As a result, simulation in software has also swept the distributed systems community as well. Before pushing out your multi-node communication fabric, you should probably simulate it extensivley to test for fault tolerance. Simulating multiple processors existing and interacting is truly a sight to see.

Robotics and the physical world, simulation has seen a great benefit as well. Do you think people would test the changes they make on their autonomus vehicles by pushing them out in the world and hoping everything goes right? Do you think people test the changes they make to their surgical robots before operating on a human? Absolutely not! Simulators are the core focus of this R&D.

Once you realise this, you can't go back. Everything you see in the world, you realise there was some form of a simulation in software that likely went behind that product.

Simulators save time, money, regrets, and open up a world full of opportunities where you can make the optimal choice for something that doesn't even exist yet. I hypothesis as software gets easier to write, and hardware for vastly different applications gets easier to make, the need for simulators will only increase.