# Experiment-4 BFS

Sequential and Parallel Breadth-First Search (BFS) using CUDA.

## Compiling

Before compiling, you need to change working directory to BFS.
```bash
cd experiment-4/bfs
```

To compile source code into binary:
```bash
make -C src all
```

To remove the compiled executable `bfs-exec`:
```bash
make -C src clean 
```

## Running

As this program executes BFS in CPU and then in CUDA sequentially without specifying any parameters, providing the parameters below only is OK.

```bash
./bfs-exec <start vertex> <number of vertices> <number of edges>
```

## Performance Testing

Simply run `test.sh`, which includes all the needed parameters be given.

Below is a recommended way to run `test.sh` to test the performance.

```bash
./test.sh | tee log.txt
```