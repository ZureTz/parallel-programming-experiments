#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "cg.h"

// Solve Ax = b for x, using the Conjugate Gradient method.
// Terminates once the maximum number of steps or tolerance has been reached
double *conjugate_gradient_serial(double *A, double *b, int N, int max_steps,
                                  double tol) {
  // 分配解向量x的内存，初始值为0
  double *x = malloc(N * sizeof(double));
  memset(x, 0, N * sizeof(double)); // x := 0

  // 分配残差向量r、搜索方向向量p和矩阵-向量乘积Ap的内存
  double *r = malloc(N * sizeof(double));
  double *p = malloc(N * sizeof(double));
  double *Ap = malloc(N * sizeof(double));

  // 初始化残差r为b
  memcpy(r, b, N * sizeof(double));

  // 设置搜索方向p为残差r
  memcpy(p, r, N * sizeof(double));

  double rho_old = 0.0, rho_new = 0.0, alpha = 0.0, beta = 0.0;
  double tol_squared = tol * tol; // 计算容忍度的平方

  // 计算初始的rho_old（r的范数平方）
  for (int i = 0; i < N; i++)
    rho_old += r[i] * r[i];

  // 开始迭代，最多进行max_steps次
  for (int k = 0; k < max_steps; k++) {
    // 检查是否达到容忍度，如果rho_old小于等于tol_squared则提前退出
    if (rho_old <= tol_squared)
      break;

    // 计算Ap = A * p
    for (int i = 0; i < N; i++) {
      Ap[i] = 0.0; // 初始化Ap
      for (int j = 0; j < N; j++)
        Ap[i] += A[i * N + j] * p[j]; // 计算矩阵-向量乘积
    }

    // 计算p的转置和Ap的乘积pAp
    double pAp = 0.0;
    for (int i = 0; i < N; i++)
      pAp += p[i] * Ap[i];

    // 计算步长α
    alpha = rho_old / pAp;

    // 更新解向量x = x + α * p
    for (int i = 0; i < N; i++)
      x[i] += alpha * p[i];

    // 更新残差r = r - α * Ap
    for (int i = 0; i < N; i++)
      r[i] -= alpha * Ap[i];

    // 计算新的rho_new（r的范数平方）
    rho_new = 0.0;
    for (int i = 0; i < N; i++)
      rho_new += r[i] * r[i];

    // 检查是否达到容忍度，如果rho_new小于等于tol_squared则提前退出
    if (rho_new <= tol_squared)
      break;

    // 计算新的搜索方向的系数β
    beta = rho_new / rho_old;

    // 更新搜索方向p = r + β * p
    for (int i = 0; i < N; i++)
      p[i] = r[i] + beta * p[i];

    // 更新rho_old为当前的rho_new
    rho_old = rho_new;
  }

  // 释放分配的内存
  free(r);
  free(p);
  free(Ap);

  // 返回最终的解向量x
  return x;
}

void conjugate_gradient_parallel(process_data row, equation_data equation,
                                 int N, int max_steps, double tol) {
  int step;
  double alpha, beta, r_norm, r_norm_prev, pAp, local_r_norm, local_pAp;
  double *r, *p, *Ap, *global_p;

  // 为向量分配内存
  r = (double *)malloc(row.count * sizeof(double));  // 残差向量 r
  p = (double *)malloc(row.count * sizeof(double));  // 搜索方向向量 p
  Ap = (double *)malloc(row.count * sizeof(double)); // A * p
  global_p = (double *)malloc(N * sizeof(double));   // 全局向量 p

  // 初始化 r = b - A * x, p = r
  memcpy(r, equation.b, row.count * sizeof(double)); // 将b复制给r
  memset(Ap, 0, row.count * sizeof(double));         // 将Ap初始化为零
  memset(equation.x_star, 0, row.count * sizeof(double)); // 初始化 x_star = 0

  // 初始化 p = r
  memcpy(p, equation.b, row.count * sizeof(double));

  // 计算初始的 r_norm = r^T * r（局部和全局归约）
  local_r_norm = 0.0;
  for (int i = 0; i < row.count; ++i)
    local_r_norm += r[i] * r[i]; // 计算局部残差的平方和
  MPI_Allreduce(&local_r_norm, &r_norm, 1, MPI_DOUBLE, MPI_SUM,
                row.comm); // 全局归约计算 r_norm

  // 如果初始残差范数非常小，则提前退出
  if (r_norm <= tol * tol) {
    free(r);
    free(p);
    free(Ap);
    free(global_p);
    return;
  }

  // 主共轭梯度循环
  for (step = 0; step < max_steps && r_norm > tol * tol; ++step) {
    // Gather all local p[] to global_p for matrix-vector multiplication
    MPI_Allgatherv(p, row.count, MPI_DOUBLE, global_p, row.counts, row.displs,
                   MPI_DOUBLE, row.comm);

    // 矩阵-向量乘法 Ap = A * p
    for (int i = 0; i < row.count; ++i) {
      Ap[i] = 0;
      for (int j = 0; j < row.N; ++j) {
        Ap[i] += equation.A[i * row.N + j] * global_p[j]; // 使用全局的 p
      }
    }

    // 计算 pAp = p^T * A * p（局部和全局归约）
    local_pAp = 0.0;
    for (int i = 0; i < row.count; ++i)
      local_pAp += p[i] * Ap[i]; // 计算局部 p 和 Ap 的内积
    MPI_Allreduce(&local_pAp, &pAp, 1, MPI_DOUBLE, MPI_SUM,
                  row.comm); // 全局归约计算

    // 检查 pAp 是否为0，以避免除以零
    if (pAp == 0.0)
      break;

    // 计算步长 alpha = r^T * r / p^T * A * p
    alpha = r_norm / pAp;

    // 更新 x_star = x_star + alpha * p
    for (int i = 0; i < row.count; ++i)
      equation.x_star[i] += alpha * p[i]; // 更新解向量

    // 更新 r = r - alpha * A * p
    for (int i = 0; i < row.count; ++i)
      r[i] -= alpha * Ap[i]; // 更新残差向量

    // 计算新的残差范数 r_norm
    local_r_norm = 0.0;
    for (int i = 0; i < row.count; ++i)
      local_r_norm += r[i] * r[i]; // 计算局部残差的平方和
    r_norm_prev = r_norm;          // 保存前一次的r_norm
    MPI_Allreduce(&local_r_norm, &r_norm, 1, MPI_DOUBLE, MPI_SUM,
                  row.comm); // 全局归约计算

    // 如果 r_norm 非常小，则提前退出
    if (r_norm <= tol * tol)
      break;

    // 计算新的搜索方向 p = r + beta * p
    beta = r_norm / r_norm_prev; // 计算更新系数
    for (int i = 0; i < row.count; ++i)
      p[i] = r[i] + beta * p[i]; // 更新搜索方向
  }
  free(r);
  free(p);
  free(Ap);
  free(global_p);
}
