A command line app for Structure-from-Motion 3d estimation using OpenCV and Eigen. HEAVILY based off of https://github.com/royshil/SfM-Toy-Library/tree/master/SfMToyLib.

Requirements 
============
make and associated build tools, 
g++ or clang, 
OpenCV, 
Eigen3, 
pkg-config, 
A PCD (Point-Cloud Library format) viewer.

Building
========
Set the EIGEN\_PATH variable in the top-level Makefile properly, then run `make` from the top-level directory.

Running a demo
==============
`./sfm dataset/crazyhorse RICH CPU 0.5` 
