# SiMPL for multi-material topology optimization

This repository contains the code for the numerical examples of

> Peter Gangl, Brendan Keith, Dohyun Kim, Boyan S. Lazarov and
> Thomas M. Surowiec,
> [*The SiMPL Method for Density-Based Multi-Material Topology Optimization*](https://doi.org/10.48550/arXiv.2605.11994)

The examples are written with [MFEM](https://mfem.org) and use the Simplicial
Mirror descent with a Projected Latent variable (SiMPL) method: mirror descent
with the Gibbs entropy, a Bregman projection onto the volume constraints, and
a generalized Barzilai-Borwein step size with Armijo backtracking.

| Paper | Executable | Description |
|---|---|---|
| Section 6.1 | `cantilever` | void and three isotropic materials, Bregman-Dykstra projection with Anderson acceleration |
| Section 6.2 | `orthotropic` | void and one orthotropic material with 8 orientations |

The default options of each executable reproduce the corresponding example.

## Requirements

- MFEM 4.10 built with CMake and MPI (`MFEM_USE_MPI=YES`).
- A C++17 compiler and MPI.

## Build

By default, MFEM is expected in `../mfem/build`. Otherwise, create a file
`user.cmake` in this directory with

```cmake
set(MFEM_RELEASE_DIR "/path/to/mfem/build")
set(MFEM_DEBUG_DIR "/path/to/mfem/build-debug")
```

and build with

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
```

The executables are in `build/src`.

## Run

```bash
cd build/src
mpirun -np 8 ./cantilever
mpirun -np 8 ./orthotropic
```

Each run prints one row per iteration: the iteration number, the objective,
the gap residual, the step size and the number of backtracking steps. Use
`-pv` to save ParaView output in `ParaView/`, `-vis` to send the fields to a
running GLVis server, and `-h` to list all options.

## Layout

- `include/`, `lib/`: Legendre functions (`legendre`), forward-mode automatic
  differentiation (`ad`), elastic materials and the PDE filter
  (`elasticity`), boundary conditions, solvers, coefficients and output
  (`bdr`, `solvers`, `cf`, `logger`).
- `src/`: one source file per executable.
