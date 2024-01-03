# Simulation Server

The server app for the Blood Simulation project, available for Linux, made in CUDA 11.8 and C++ 17.
Our primary intention is for it to be launched on HPC servers.

## Authors
 * Hubert Lawenda
 * Piotr Nieciecki
 * Filip Skrzeczkowski

## Prerequisites
The poject requires **CUDA 11.8** installed and an Nvidia GPU with compute capability **>=5.0**.

When it comes to libraries, most of the project is self-contained.
The only thing you have to do beforehand is to install the **gstreamer-1.0 1.22.8** library (`apt install gstreamer` or you package manager's equivalent).
**X11** is only required if **WINDOW_RENDERING** is defined. Otherwise the app can run perfectly fine on a headless server.

## Build
The primary method of building the solution is to use cmake.
Typically you'd want to create a directory named "build" at the level of CMakeLists.txt, enter it, and execute:

`cmake ..`

`make`

The src/config directory contains files that can be overwritten by the client app in order to change the configuration of the simulation. Recompilation is necessary after each such change.

PLease note that because of the use of metaprogramming the compilation time may be pretty long.

## Usage

Upon launching the program, it will immediatelly begin listening for a client connection. If such connection is established successfully, the app will begin simulating the movement of blood cells inside a vein on GPU, render it (offscreen if **WINDOW_RENDERING** is not defined) and stream it back to the client.
