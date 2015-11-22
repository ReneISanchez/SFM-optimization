A small app that runs the Structure-from-Motion algorithm and display the results. Not a real GUI, but opens a window to display the final results.
This might evolve into a (slightly) more functional GUI if there is an easy way to integrate Qt or similar library, and resolve some configuration/dependencies issues (i.e. problems with VTK version shipped with Ubuntu 14.04)

Requirements
============
The SfM library (libsfm.so) and associated dependencies (OpenCV, eigen, etc.)
PCL library and all associated dependencies (cmake, boost, eigen, etc.)


Building
========
`mkdir build`
`cd build`
`cmake -DCMAKE_BUILD_TYPE=Release ..`
`make`

Note that you can specify the path to the top-level SfM folder with the *SFM_PATH* variable.
You might need to specify the PCL location too.
I recommend _ccmake_ or _cmake-gui_ to edit the variables.


Running a demo
==============
(From your build directory)
`./sfm-gui ../../dataset/crazyhorse RICH CPU 0.5`
