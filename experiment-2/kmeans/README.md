# Experiment 2 - KMeans Clustering

## Build 

### Prerequisites

- OS: GNU/Linux
- Compiler: gcc, with OpenMP support

### Build Steps

Execute the following commands to build:

```bash
cd experiment-2/kmeans/
make -C src all
```

After `make`, there will be two executable files compiled. Their usage will be covered below.

## Running Executables

First, you need to change your working directory to the source code directory.

```bash
cd experiment-2/kmeans/src
```

### For Serial Version

```bash
./kmeans-serial -s <number objects> -k <number clusters> [ -t <threshold> ]
```

You can use parameter `-s` to specify the number of points, also use `-k` to specify how many how many clusters will be simulated. The `-t` parameter is optional, which represent the precision. (Default `0.001`) If given a smaller threshold, the precision will be higher but it will take more time to run.

### For Parallel Version

```bash
./kmeans-parallel -s <number objects> -k <number clusters> -n <number threads> [ -t <threshold> ]
```

You can use parameter `-s` to specify the number of points, also use `-k` to specify how many how many clusters will be simulated. The `-t` parameter is optional, which represent the precision. (Default `0.001`) If given a smaller threshold, the precision will be higher but it will take more time to run. Also, you must specify how many threads you want the program to run using parameter `-n`.

## Performance Test

The procedure of performance test is packed into a bash script. You can run this script directly without any compiling, as this script will do this automatically. 

```bash
cd experiment-2/kmeans
chmod +x ./test.sh && ./test.sh
```

If you want store the output of the `test.sh` as log, you can execute the command below instead.

```bash] 
./test.sh | tee test.log
```

