## General

Sequential and Parallel K-nearest Neighbors (knn) using CUDA.


## Compilation

`make clean` removes the compiled executable `knn-exec`.

`make` compiles the code.

## Running

`./knn_exec <k nearest neighbors> <number of classes> <number of existing samples> <number of new samples>`.

编译和执行的方式不变
build.sh里面有所有的情况
knnCPU.cpp里面有两种方法，一种是寻找最近邻时是参考小根堆实现，另一种直接k次循环，每次遍历距离数组，找到k个最小的，若要实现可选择性注释一个函数