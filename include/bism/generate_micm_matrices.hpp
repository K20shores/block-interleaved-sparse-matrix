#include <micm/util/sparse_matrix.hpp>
#include <micm/util/sparse_matrix_vector_ordering.hpp>

using MicmSparseMatrix = micm::SparseMatrix<double>;

MicmSparseMatrix create_sparse_matrix(int num_species, int num_cells);