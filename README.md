# Simulation Server

The server app for the Blood Simulation project, available for Linux, made in CUDA 11.8 and C++ 17.
Our primary intention is for it to be launched on a HPC cluster.

## Authors
 * Hubert Lawenda
 * Piotr Nieciecki
 * Filip Skrzeczkowski

## Prerequisites
The poject requires **CUDA 11.8** installed and an Nvidia GPU with compute capability **>=5.0**.

In order to make the building process easier, the "Containers" directory contains a Singularity container.
**X11** is only required if `WINDOW_RENDERING` is defined (see later). Otherwise the app can run perfectly fine on a headless server.

## Build
The primary method of building the solution is to use cmake.
Typically you'd want to create a directory named "build" at the level of CMakeLists.txt, enter it, and execute:

In order to run the container execute the following commands:
  1. `export NVIDIA\_DRIVER\_CAPABILITIES=compute,utility,graphics`
  2. `singularity shell --nv --nvccli -B /usr/share/glvnd <container_location>`
  3. `cmake <flags> ..`
  4. `make`

The flags mentioned above are as follows:

 * `-DWINDOW_RENDERING` - if switched on, the server will create a graphical window of its own instead of streaming the data to the client
 * `-DMULTI_GPU` - if switched on, the server will use multiple GPUs to calculate the simulation

The src/config directory contains files that can be overwritten by the client app in order to change the configuration of the simulation. Recompilation is necessary after each such change.

PLease note that because of the use of metaprogramming the compilation time may be very long.

## Usage

Upon launching the program, it will immediatelly begin listening for a client connection. If such connection is established successfully, the app will begin simulating the movement of blood cells inside a vein on GPU, render it (offscreen if `WINDOW_RENDERING` is not defined) and stream it back to the client.
