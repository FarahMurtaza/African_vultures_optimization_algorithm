# African Vultures Optimization Algorithm (AVOA) Implementation in MPI

This repository contains an implementation of the African Vulture Optimization Algorithm (AVOA) using the Message Passing Interface (MPI) for parallel processing. The project is written in C++ and utilizes Boost libraries for mathematical computations.

## Table of Contents

- [Introduction](#introduction)
- [Algorithm](#algorithm)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Conclusion](#conclusion)

## Introduction

The African Vulture Optimization Algorithm (AVOA) is a nature-inspired optimization technique that mimics the behavior of vultures searching for food. This algorithm is particularly useful for solving complex optimization problems with multiple local optima.

This implementation leverages MPI to distribute the computational load across multiple processors, making it suitable for large-scale optimization problems.

## Algorithm

The AVOA algorithm follows these steps:

1. **Initialization**: Generate an initial population of vultures with random positions.
2. **Fitness Calculation**: Evaluate the fitness of each vulture based on the problem-specific objective function.
3. **Update Positions**: Update the positions of vultures using several mathematical equations that simulate the vultures' adaptive behavior.
4. **Best Solution Update**: Keep track of the best and second-best solutions found so far.
5. **Iterations**: Repeat the fitness calculation and position update steps for a specified number of iterations or until a convergence criterion is met.

## Getting Started

Before running the AVOA algorithm, ensure that you have the following prerequisites installed on your system:

- C and C++ Compiler: A compiler that supports C++14 standard.
- MPI Library: An implementation of the Message Passing Interface (MPI), such as MPICH or OpenMPI.
- Boost Libraries: Boost C++ libraries, particularly for mathematical functions.


### Installation

To install and run this project, follow these steps:

- Clone the repository: `git clone https://github.com/FarahMurtaza/African_vultures_optimization_algorithm`
- Navigate to the project directory: `cd African_vultures_optimization_algorithm`
- Build the project: `make`
- Run the executable: `./main`


## Conclusion

Thank you for exploring our project! We hope you find it valuable and easy to use. If you encounter any issues, have suggestions for improvements, or would like to contribute, please feel free reach out.

Your feedback is highly appreciated!
