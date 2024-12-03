# Experiment 2 - Ocean Simulation

## Build 

### Prerequisites

- OS: GNU/Linux
- Compiler: gcc, with OpenMP support

### Build Steps

Execute the following commands to build:

```bash
cd experiment-2/ocean-simulation/
make -C src all
```

After `make`, there will be two executable files compiled. Their usage will be covered below.

## Running Executables

First, you need to change your working directory to the source code directory.

```bash
cd experiment-2/ocean-simulation/src
```

### For Serial Version

```bash
./serial_ocean -d <dimension> -t <timesteps>
```

You can use parameter `-d` to specify the dimension of the ocean to simulate, also use `-t` to specify how many frames you want to simulate.

### For Parallel Version

```bash
./omp_ocean -d <dimension> -t <timesteps> -n <threads>
```

You can use parameter `-d` to specify the dimension of the ocean to simulate, also use `-t` to specify how many frames you want to simulate. Also, you must specify how many threads you want the program to run using parameter `-n`.

## Performance Test

The procedure of performance test is packed into a bash script. You can run this script directly without any compiling, as this script will do this automatically. 

```bash
cd experiment-2/ocean-simulation
chmod +x ./test.sh && ./test.sh
```

If you want store the output of the `test.sh` as log, you can execute the command below instead.

```bash] 
./test.sh | tee test.log
```

