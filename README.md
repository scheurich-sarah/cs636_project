Neither of my attempts to blend approaches beginning from either repo work. No need to try and run anything. I waivered multiple times on where to start and never got a python or a kernelized approach working.

The pytorch-only implementation (files with _paper) that I found required a lot of support files and overhead.

The Graphpy implementation (files with _graphpy) was leading me to feel like I needed to kernelize everything and was too hard for me to follow.

repository for CSCI 636 final project files

## Compilation of Kernel: At the top level directory
```
mkdir build
cd build
cmake ../kernel
make

Copy the binary files into the directory where you'll execute the python script

