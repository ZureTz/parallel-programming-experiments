#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "body.h"
MPI_Datatype vect_mpi_t;   // Derived data type used
const double G = 6.67e-11; // Gravitational constant
int my_rank, comm_sz;      // Process ID and total number of processes
vect_t *vel = NULL;        // Global speed, used for outputs of process 0

void Usage(char *prog_name) // Input Description
{
  fprintf(stderr, "usage: mpiexec -n <nProcesses> %s\n", prog_name);
  fprintf(stderr, "<nParticle> <nTimestep> <sizeTimestep>\n");
  exit(0);
}

void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p,
              double *delta_t_p) // get parameter information
{
  if (my_rank == 0) {
    if (argc != 4)
      Usage(argv[0]);
    *n_p = strtol(argv[1], NULL, 10);
    *n_steps_p = strtol(argv[2], NULL, 10);
    *delta_t_p = strtod(argv[3], NULL);
    if (*n_p <= 0 || *n_p % comm_sz || *n_steps_p < 0 ||
        *delta_t_p <= 0) // illegal inputs
    {
      printf("illegal input\n");
      if (my_rank == 0)
        Usage(argv[0]);
      MPI_Finalize();
      exit(0);
    }
  }
  MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(n_steps_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(delta_t_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// the initial conditions are automatically generated, and all processes call
// this function because of gather communication
void Gen_init_cond(double masses[], vect_t pos[], vect_t loc_vel[], int n,
                   int loc_n) {
  const double mass = 10000, gap = 0.01, speed = 0;
  if (my_rank == 0) {
    int ny = ceil(sqrt(n));
    double y = 0.0;
    for (int i = 0; i < n; i++) {
      masses[i] = mass;
      pos[i][X] = i * gap;
      if ((i + 1) % ny == 0)
        ++y;
      pos[i][Y] = y;
      vel[i][X] = 0.0;
      vel[i][Y] = 0.0;
    }
  }
  // Synchronization mass, position information, distribution vector information
  MPI_Bcast(masses, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(pos, n, vect_mpi_t, 0, MPI_COMM_WORLD);
  MPI_Scatter(vel, loc_n, vect_mpi_t, loc_vel, loc_n, vect_mpi_t, 0,
              MPI_COMM_WORLD);
}

void Output_parallel(double masses[], vect_t pos[], vect_t loc_vel[], int n,
                     int loc_n) // output current states
{
  // gather information from all processes for outputs
  MPI_Gather(loc_vel, loc_n, vect_mpi_t, vel, loc_n, vect_mpi_t, 0,
             MPI_COMM_WORLD);
  if (my_rank == 0) {
    printf("Output Parallel State\n");
    for (int i = 0; i < n; i++)
      printf(" %3d X: %.5f Y: %.5f\n", i, pos[i][X], pos[i][Y]);
    printf("\n");
    fflush(stdout);
  }
}

void Output_serial(double masses[], vect_t pos[], vect_t vel[],
                   int n) // output current states
{
  if (my_rank == 0) {
    printf("Output Serial State\n");
    for (int i = 0; i < n; i++)
      printf(" %3d X: %.5f Y: %.5f\n", i, pos[i][X], pos[i][Y]);
    printf("\n");
    fflush(stdout);
  }
}

// nbody parallel implementation
void nbody_parallel(double masses[], vect_t loc_forces[], vect_t pos[],
                    vect_t loc_pos[], vect_t loc_vel[], int n, int loc_n,
                    int n_steps, double delta_t) {

  vect_t *global_pos =
      (vect_t *)malloc(n * sizeof(vect_t)); // Allocate global position array

  for (int step = 1; step <= n_steps; step++) {
    // Gather global positions from all processes
    MPI_Allgather(loc_pos, loc_n, vect_mpi_t, global_pos, loc_n, vect_mpi_t,
                  MPI_COMM_WORLD);

    // Initialize local forces to zero
    for (int i = 0; i < loc_n; i++) {
      loc_forces[i][X] = 0.0;
      loc_forces[i][Y] = 0.0;
    }

    // Compute pairwise forces between particles
    for (int i = 0; i < loc_n; i++) {
      for (int j = 0; j < n; j++) {
        if (my_rank * loc_n + i != j) {
          // Calculate the distance between particles
          const double dx = global_pos[j][X] - loc_pos[i][X];
          const double dy = global_pos[j][Y] - loc_pos[i][Y];

          const double dist_squared = dx * dx + dy * dy;
          const double dist = sqrt(dist_squared);

          if (dist_squared == 0) {
            continue;
          }

          // Compute force magnitude
          const double f =
              (G * masses[my_rank * loc_n + i] * masses[j]) / dist_squared;

          vect_t force;

          // Apply force components
          force[X] = f * dx / dist;
          force[Y] = f * dy / dist;

          // Accumulate force to local forces
          loc_forces[i][X] += force[X];
          loc_forces[i][Y] += force[Y];
        }
      }
    }

    // Update velocities and positions based on forces
    for (int i = 0; i < loc_n; i++) {
      double ax = loc_forces[i][X] / masses[my_rank * loc_n + i];
      double ay = loc_forces[i][Y] / masses[my_rank * loc_n + i];

      loc_vel[i][X] += delta_t * ax;
      loc_vel[i][Y] += delta_t * ay;

      loc_pos[i][X] += delta_t * loc_vel[i][X];
      loc_pos[i][Y] += delta_t * loc_vel[i][Y];
    }

#ifdef DEBUG
    if (step == n_steps)
      Output_parallel(masses, global_pos, loc_vel, n, loc_n);
#endif
  }

  free(global_pos); // Free allocated memory for global positions
}

// nbody serial implementation
void nbody_serial(double masses[], vect_t forces[], vect_t pos[], vect_t vel[],
                  int n, int n_steps, double delta_t) {

  for (int step = 1; step <= n_steps; step++) {
    // 初始化所有粒子的合力为 0
    for (int i = 0; i < n; i++) {
      forces[i][X] = 0.0;
      forces[i][Y] = 0.0;
    }

    // 计算所有粒子之间的引力并累加到每个粒子的合力上
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) // 对称性优化
      {
        // 计算粒子 i 和 j 之间的距离
        const double dx = pos[j][X] - pos[i][X];
        const double dy = pos[j][Y] - pos[i][Y];
        // 粒子间距离的平方
        const double dist_squared = dx * dx + dy * dy;
        // 粒子间距离
        const double dist = sqrt(dist_squared);

        if (dist_squared == 0) {
          continue;
        }

        // 计算引力大小
        const double f = (G * masses[i] * masses[j]) / dist_squared;

        // 用于临时存储单对粒子的力
        vect_t force;

        // 将力分解到 x 和 y 方向
        force[X] = f * (dx / dist);
        force[Y] = f * (dy / dist);

        // 叠加力到粒子 i 和 j 的合力上（对称处理）
        forces[i][X] += force[X];
        forces[i][Y] += force[Y];
        forces[j][X] -= force[X]; // 对称的反向力
        forces[j][Y] -= force[Y];
      }
    }

    // 更新所有粒子的速度和位置，使用跨越式积分法
    for (int i = 0; i < n; i++) {
      // 根据合力计算加速度
      double ax = forces[i][X] / masses[i];
      double ay = forces[i][Y] / masses[i];

      // 更新速度：v = v + 0.5 * Δt * a (半步更新速度)
      vel[i][X] += 0.5 * delta_t * ax;
      vel[i][Y] += 0.5 * delta_t * ay;

      // 更新位置：r = r + Δt * v (根据半步更新后的速度更新位置)
      pos[i][X] += delta_t * vel[i][X];
      pos[i][Y] += delta_t * vel[i][Y];

      // 重新计算加速度，因为位置已经更新
      ax = forces[i][X] / masses[i];
      ay = forces[i][Y] / masses[i];

      // 再次更新速度：v = v + 0.5 * Δt * a (跨越式完成速度更新)
      vel[i][X] += 0.5 * delta_t * ax;
      vel[i][Y] += 0.5 * delta_t * ay;
    }

#ifdef DEBUG
    if (step == n_steps)
      Output_serial(masses, pos, vel, n);
#endif
  }
}

int main(int argc, char *argv[]) {
  int n, loc_n; // Number of particles, number of particles per process, current
                // particle (loop variable)
  int n_steps;  // Number of steps, current step
  double delta_t;                    // Step length
  double *masses;                    // Masses of particles
  vect_t *pos, *loc_pos;             // Positions of particles
  vect_t *loc_vel;                   // Speeds of particles in each process
  vect_t *forces, *loc_forces;       // Gravities of particles
  double start, finish;              // Timer
  double parallel_time, serial_time; // parallel and serial duration

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Type_contiguous(DIM, MPI_DOUBLE, &vect_mpi_t); // Derivation type required
  MPI_Type_commit(&vect_mpi_t);

  // Get parameters and initialize arrays
  Get_args(argc, argv, &n, &n_steps, &delta_t);
  loc_n = n / comm_sz; // n % comm_sz == 0
  masses = (double *)malloc(n * sizeof(double));
  pos = (vect_t *)malloc(n * sizeof(vect_t));
  loc_pos = pos + my_rank * loc_n;
  loc_vel = (vect_t *)malloc(loc_n * sizeof(vect_t));
  loc_forces = (vect_t *)malloc(loc_n * sizeof(vect_t));
  if (my_rank == 0) {
    vel = (vect_t *)malloc(n * sizeof(vect_t));
    forces = (vect_t *)malloc(n * sizeof(vect_t));
  }
  Gen_init_cond(masses, pos, loc_vel, n, loc_n);

  // Start calculating and timing
  if (my_rank == 0)
    start = MPI_Wtime();

  nbody_parallel(masses, loc_forces, pos, loc_pos, loc_vel, n, loc_n, n_steps,
                 delta_t);

  if (my_rank == 0) {
    parallel_time = MPI_Wtime() - start;
  }

  if (my_rank == 0) {
    start = MPI_Wtime();
    nbody_serial(masses, forces, pos, vel, n, n_steps, delta_t);
    serial_time = MPI_Wtime() - start;
  }

  // Output time
  if (my_rank == 0) {
    printf("serial time = %f s\n", serial_time);
    printf("parallel time = %f s\n", parallel_time);
    FILE *file = fopen("../data.txt", "a");
    fprintf(file, "%d %d %d %f %f %f\n", comm_sz, n, n_steps, serial_time,
            parallel_time, serial_time / parallel_time);
    fclose(file);
    free(vel);
    free(forces);
  }

  MPI_Type_free(&vect_mpi_t);
  free(masses);
  free(pos);
  free(loc_vel);
  free(loc_forces);
  MPI_Finalize();
  return 0;
}