# Experiment-3 N-body

Sequential and Parallel N-body using MPI

## Compilation

- `make` compiles the code

```bash
cd experiment-3/n-body
make -C src all
```

- `make clean` removes the compiled executable `nbody`.

```bash
make -C src clean
```

## Running

```bash
mpirun -np <nProcess> ./nbody <nParticle> <nTimestep> <sizeTimestep>
```
Where

- `<nProcess>` is the desired number of processes,

- `<nParticle>` is the number of particles,

- `<nTimestep>` is the number of time steps.

- `<sizeTimestep>` is the length of the time step.

Note that `<nParticle>` % `<nProcess>` == 0

## Performance Testing

You can use `test.sh` to test performance. Below is the recommended usage:

```bash
./test.sh | tee log.txt
```

The command runs `test.sh` to perform performance testing and uses `tee` to display the output in the terminal and save it to `log.txt` for later analysis.