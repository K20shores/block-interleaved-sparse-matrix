#include <iostream>
#include <Eigen/SparseLU>
#include <Eigen/Dense>
#include <bism/block_sparse.hpp>

void solveWithEigenLU(const Eigen::SparseMatrix<double>& A) {
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    if (solver.info() != Eigen::Success) {
        std::cerr << "LU factorization failed!" << std::endl;
        return;
    }

    Eigen::VectorXd b = Eigen::VectorXd::Ones(A.rows());
    Eigen::VectorXd x = solver.solve(b);

    std::cout << "Eigen LU Solution (first 5 elements):\n" << x.head(5) << std::endl;
}

void dense() {
    const int num_species = 3;
    const int num_cells = 3;

    // Species-major layout (RowMajor storage)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> species_major(num_species, num_cells);
    species_major << 1.1, 2.1, 3.1,
                     1.2, 2.2, 3.2,
                     1.3, 2.3, 3.3;

    std::cout << "Species-major layout (RowMajor):\n" << species_major << "\n\n";
    std::cout << "Species-major memory layout:\n";
    for (int i = 0; i < species_major.size(); ++i) {
        std::cout << species_major.data()[i] << " ";
    }
    std::cout << "\n\n";

    // Cell-major layout (ColMajor storage)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> cell_major(num_species, num_cells);
    cell_major << 1.1, 2.1, 3.1,
                  1.2, 2.2, 3.2,
                  1.3, 2.3, 3.3;

    std::cout << "Cell-major layout (ColMajor):\n" << cell_major << "\n";
    std::cout << "Cell-major memory layout:\n";
    for (int i = 0; i < cell_major.size(); ++i) {
        std::cout << cell_major.data()[i] << " ";
    }
    std::cout << "\n\n";
}

void sparse(){
    const int num_species = 3;
    const int num_cells = 3;

    // Species-major layout (RowMajor storage)
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseRowMajor;
    SparseRowMajor species_major(num_species * num_cells, num_species * num_cells);

    std::vector<Eigen::Triplet<double>> species_major_triplets;
    for (int cell = 0; cell < num_cells; ++cell) {
        for (int species = 0; species < num_species; ++species) {
            int row = species * num_cells + cell;
            int col = species * num_cells + cell;
            species_major_triplets.emplace_back(row, col, 1.0 * (species + 1) + 0.1 * (cell + 1));
        }
    }
    species_major.setFromTriplets(species_major_triplets.begin(), species_major_triplets.end());

    std::cout << "Species-major layout (RowMajor):\n";
    for (int k = 0; k < species_major.outerSize(); ++k) {
        for (SparseRowMajor::InnerIterator it(species_major, k); it; ++it) {
            std::cout << "(" << it.row() << ", " << it.col() << ") = " << it.value() << "\n";
        }
    }
    std::cout << "\n";

    std::cout << "Species-major raw memory layout:\n";
    for (int i = 0; i < species_major.nonZeros(); ++i) {
        std::cout << species_major.valuePtr()[i] << " ";
    }
    std::cout << "\n\n";

    // Cell-major layout (ColMajor storage)
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseColMajor;
    SparseColMajor cell_major(num_species * num_cells, num_species * num_cells);

    std::vector<Eigen::Triplet<double>> cell_major_triplets;
    for (int cell = 0; cell < num_cells; ++cell) {
        for (int species = 0; species < num_species; ++species) {
            int row = cell * num_species + species;
            int col = cell * num_species + species;
            cell_major_triplets.emplace_back(row, col, 1.0 * (species + 1) + 0.1 * (cell + 1));
        }
    }
    cell_major.setFromTriplets(cell_major_triplets.begin(), cell_major_triplets.end());

    std::cout << "Cell-major layout (ColMajor):\n";
    for (int k = 0; k < cell_major.outerSize(); ++k) {
        for (SparseColMajor::InnerIterator it(cell_major, k); it; ++it) {
            std::cout << "(" << it.row() << ", " << it.col() << ") = " << it.value() << "\n";
        }
    }

    std::cout << "Cell-major raw memory layout:\n";
    for (int i = 0; i < cell_major.nonZeros(); ++i) {
        std::cout << cell_major.valuePtr()[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    dense();
    sparse();

    // int numBlocks = 4, blockSize = 4;
    // BlockSparseMatrix blockSparse(numBlocks, blockSize);
    // blockSparse.generateMatrix();
    // const auto& A = blockSparse.getMatrix();
    // std::cout << "BlockSparseMatrix:\n" << A << "\n\n";

    // for (int i = 0; i < A.size(); ++i) {
    //     std::cout << A.data()[i] << " ";
    // }

    // std::cout << "Solving with Eigen...\n";
    // solveWithEigenLU(A);

    return 0;
}
