# African Vultures Optimization Algorithm (AVOA) Implementation in MPI

This repository contains an implementation of the Adaptive Vulture Optimization Algorithm (AVOA) using the Message Passing Interface (MPI) for parallel processing. The project is written in C++ and utilizes Boost libraries for mathematical computations.

## Table of Contents

- [Introduction](#introduction)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Adaptive Vulture Optimization Algorithm (AVOA) is a nature-inspired optimization technique that mimics the behavior of vultures searching for food. This algorithm is particularly useful for solving complex optimization problems with multiple local optima.

This implementation leverages MPI to distribute the computational load across multiple processors, making it suitable for large-scale optimization problems.

## Algorithm

The AVOA algorithm follows these steps:

1. **Initialization**: Generate an initial population of vultures with random positions.
2. **Fitness Calculation**: Evaluate the fitness of each vulture based on the problem-specific objective function.
3. **Update Positions**: Update the positions of vultures using several mathematical equations that simulate the vultures' adaptive behavior.
4. **Best Solution Update**: Keep track of the best and second-best solutions found so far.
5. **Iterations**: Repeat the fitness calculation and position update steps for a specified number of iterations or until a convergence criterion is met.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/FarahMurtaza/avoa-mpi.git
    cd avoa-mpi
    ```

2. **Install MPI**:
    Follow the instructions to install MPI on your system. For example, on Ubuntu:
    ```bash
    sudo apt-get update
    sudo apt-get install mpich
    ```

3. **Install Boost libraries**:
    Follow the instructions to install Boost on your system. For example, on Ubuntu:
    ```bash
    sudo apt-get install libboost-all-dev
    ```

4. **Compile the project**:
    ```bash
    make
    ```

## Usage

1. **Run the AVOA algorithm**:
    ```bash
    mpirun -np <number_of_processes> ./main
    ```

    Replace `<number_of_processes>` with the desired number of processes. For example:
    ```bash
    mpirun -np 4 ./main
    ```

## Examples

Here is an example of running the algorithm with 4 processes:
```bash
mpirun -np 4 ./main
