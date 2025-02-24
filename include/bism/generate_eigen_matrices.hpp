#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <utility>

using DenseRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DenseColMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using SparseRowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SparseColMajor = Eigen::SparseMatrix<double, Eigen::ColMajor>;

std::pair<DenseRowMajor, DenseColMajor> create_dense_matrices(int num_species, int num_cells);
std::pair<SparseRowMajor, SparseColMajor> create_sparse_matrices(int num_species, int num_cells);