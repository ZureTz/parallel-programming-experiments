# Experiment-4 KNN

Sequential and Parallel K-nearest Neighbors (KNN) using CUDA.

## Compiling

Before compiling, you need to change working directory to KNN.
```bash
cd experiment-4/knn
```

To compile source code into binary:
```bash
make -C src all
```

To remove the compiled executable `knn-exec`:
```bash
make -C src clean 
```
## Running

As this program executes KNN in CPU and then in CUDA sequentially without specifying any parameters, providing the parameters below only is OK.

```bash
./knn_exec <k nearest neighbors> <number of classes> <number of existing samples> <number of new samples>
```

## Performance testing

Simply run `test.sh`, which includes all the needed parameters be given.

Below is a recommended way to run `test.sh` to test the performance.

```bash
./test.sh | tee log.txt
```
