## General

Sequential and Parallel Breadth-First Search (BFS) using CUDA.


## Compilation

`make clean` removes the compiled executable `bfs-exec`.

`make` compiles the code.

## Running

As this program executes BFS in CPU and then in CUDA without specifying any parameters, providing the parameters below only is OK.

```bash
./bfs-exec <start vertex> <number of vertices> <number of edges>
```

## Performance Testing

Simply run `build.sh`, which includes all the needed parameters be given.

```bash
./test.sh |
```