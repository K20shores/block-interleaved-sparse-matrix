#include <iostream>
#include <Eigen/SparseLU>
#include <Eigen/Dense>
#include <bism/block_sparse.hpp>
#include <map>
#include <chrono>

#ifdef BISM_ENABLE_MICM
#include <bism/generate_micm_matrices.hpp>
#endif

void solveWithEigenLU(const Eigen::SparseMatrix<double> &A)
{
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    if (solver.info() != Eigen::Success)
    {
        std::cerr << "LU factorization failed!" << std::endl;
        return;
    }

    Eigen::VectorXd b = Eigen::VectorXd::Ones(A.rows());
    Eigen::VectorXd x = solver.solve(b);

    std::cout << "Eigen LU Solution (first 5 elements):\n"
              << x.head(5) << std::endl;
}

void print_dense(const DenseRowMajor &species_major, const DenseColMajor &cell_major)
{
    std::cout << "Species-major layout (RowMajor):\n"
              << species_major << "\n\n";
    std::cout << "Species-major memory layout:\n";
    for (int i = 0; i < species_major.size(); ++i)
    {
        std::cout << species_major.data()[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Cell-major layout (ColMajor):\n"
              << cell_major << "\n\n";
    std::cout << "Cell-major memory layout:\n";
    for (int i = 0; i < cell_major.size(); ++i)
    {
        std::cout << cell_major.data()[i] << " ";
    }
    std::cout << "\n\n";
}

void print_sparse(const SparseRowMajor &species_major, const SparseColMajor &cell_major)
{
    std::cout << "Species-major layout (RowMajor):\n";
    std::cout << species_major << "\n";
    for (int k = 0; k < species_major.outerSize(); ++k)
    {
        for (SparseRowMajor::InnerIterator it(species_major, k); it; ++it)
        {
            std::cout << "(" << it.row() << ", " << it.col() << ") = " << it.value() << "\n";
        }
    }
    std::cout << "\n";

    std::cout << "Species-major raw memory layout:\n";
    for (int i = 0; i < species_major.nonZeros(); ++i)
    {
        std::cout << species_major.valuePtr()[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Cell-major layout (ColMajor):\n";
    for (int k = 0; k < cell_major.outerSize(); ++k)
    {
        for (SparseColMajor::InnerIterator it(cell_major, k); it; ++it)
        {
            std::cout << "(" << it.row() << ", " << it.col() << ") = " << it.value() << "\n";
        }
    }

    std::cout << "Cell-major raw memory layout:\n";
    for (int i = 0; i < cell_major.nonZeros(); ++i)
    {
        std::cout << cell_major.valuePtr()[i] << " ";
    }
    std::cout << "\n";
}

int main()
{
    // make a map of strings to chrono timings
    std::map<std::string, std::chrono::duration<double>> timings;

    auto [dense_species_major, dense_cell_major] = create_dense_matrices(3, 4);
    auto [sparse_species_major, sparse_cell_major] = create_sparse_matrices(3, 4);
    print_dense(dense_species_major, dense_cell_major);
    print_sparse(sparse_species_major, sparse_cell_major);

    #ifdef BISM_ENABLE_MICM
    #endif

    return 0;
}
