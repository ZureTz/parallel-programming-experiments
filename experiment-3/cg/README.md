# Experiment-3 CG

Sequential and Parallel Conjugate Gradient method using MPI

## Compilation

- `make` compiles the code

```bash
cd experiment-3/cg
make -C src all
```

- `make clean` removes the compiled executable `cg`.

```bash
make -C src clean
```

## Running

```bash
mpirun -np <p> ./cg <N> <max_steps>
```

Where 
- `<p>` is the desired number of processes

- `<N>` is the problem size - i.e. the N in the NxN matrix generated

- `<max_steps>` is the max number of steps of conjugate gradient method

## Performance Testing

You can use `test.sh` to test performance. Below is the recommended usage:

```bash
./test.sh | tee log.txt
```

The command runs `test.sh` to perform performance testing and uses `tee` to display the output in the terminal and save it to `log.txt` for later analysis.
