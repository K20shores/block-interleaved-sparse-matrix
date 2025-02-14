#include <cuda_runtime.h>
#include <cusparse.h>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

void solveWithCuSparseLU(const Eigen::SparseMatrix<double>& A) {
  // Initialize cuSPARSE
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  // Convert Eigen Sparse Matrix to CSR format for cuSPARSE
  int rows = A.rows();
  int cols = A.cols();
  std::vector<int> h_csrRowPtr(rows + 1);
  std::vector<int> h_colIndices(A.nonZeros());
  std::vector<double> h_values(A.nonZeros());

  for (int k = 0; k < A.outerSize(); ++k) {
      h_csrRowPtr[k] = A.outerIndexPtr()[k];
      for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
          h_colIndices[it.index()] = it.col();
          h_values[it.index()] = it.value();
      }
  }
  h_csrRowPtr[rows] = A.nonZeros();

  // Allocate memory on GPU
  int *d_csrRowPtr, *d_colIndices;
  double *d_values, *d_b, *d_x;
  cudaMalloc((void**)&d_csrRowPtr, (rows + 1) * sizeof(int));
  cudaMalloc((void**)&d_colIndices, A.nonZeros() * sizeof(int));
  cudaMalloc((void**)&d_values, A.nonZeros() * sizeof(double));
  cudaMalloc((void**)&d_b, rows * sizeof(double));
  cudaMalloc((void**)&d_x, rows * sizeof(double));

  // Copy data to GPU
  cudaMemcpy(d_csrRowPtr, h_csrRowPtr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_colIndices, h_colIndices.data(), A.nonZeros() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, h_values.data(), A.nonZeros() * sizeof(double), cudaMemcpyHostToDevice);

  // Create cuSPARSE matrix descriptor
  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  // Solve Ax = b using cuSPARSE
  double alpha = 1.0;
  double beta = 0.0;
  cusparseDcsrlsvlu(handle, rows, A.nonZeros(), descr, d_values, d_csrRowPtr, d_colIndices, d_b, 1e-10, 0, d_x);

  // Copy solution back
  std::vector<double> h_x(rows);
  cudaMemcpy(h_x.data(), d_x, rows * sizeof(double), cudaMemcpyDeviceToHost);

  std::cout << "cuSPARSE LU Solution (first 5 elements):\n";
  for (int i = 0; i < std::min(5, rows); ++i) {
      std::cout << h_x[i] << "\n";
  }

  // Cleanup
  cudaFree(d_csrRowPtr);
  cudaFree(d_colIndices);
  cudaFree(d_values);
  cudaFree(d_b);
  cudaFree(d_x);
  cusparseDestroy(handle);
  cusparseDestroyMatDescr(descr);
}