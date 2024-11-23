# Paralleled sorting algorithms

This is experiment-1, focusing on **Parallelism for ShellSort and RadixSort utilizing Pthread**. Please read this README first before use.

## Build 

### Prerequisites
- OS: GNU/Linux
- Compiler: g++ with glibc

### Build Steps
Execute the following commands to build:
```bash
cd experiment-1/
make all
```
After `make`, there will be four executable files compiled. Their usage will be covered below.

## Usage

### Shell Sorter

Perform a Shell Sorter on a randomly generated array of a specified size and obtain its runtime.

The serial version:

```bash
./shellsort -s <array size> [ -r <seed> ] 
```

The parallel version:

```bash
./shellsort-parallel -s <array size> -n <num threads> [ -r <seed> ]
```

### Radix Sorter 


Perform a Radix Sorter on a randomly generated array of a specified size and obtain its runtime.


The serial version:

```bash
./radixsort -s <array size> [ -r <seed> ]
```

The parallel version:

```bash
./radixsort-parallel -s <array size> -n <num threads> [ -r <seed> ]
```

### Parameter Explanation

`-s`: represents the size of the random array you need to sort, `-s 10` Generate a random array of size 10.

`-n`: represents the number of threads needed to run parallel programs, `-n 32` Indicates the use of 32 threads when running parallel programs.

### Running Example

```bash
./radixsort -s 5000000
```

```bash
./shellsort-parallel -s 10000000 - n 64
```

## Test
The testing script has been written in `test.sh`. To test the performance on your computer, just run this shell script.
```bash
chmod +x ./test.sh
./test.sh
```



