#include <iostream>
#include <Eigen/SparseLU>
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


int main() {
    int numBlocks = 4, blockSize = 4;
    BlockSparseMatrix blockSparse(numBlocks, blockSize);
    blockSparse.generateMatrix();
    const auto& A = blockSparse.getMatrix();

    std::cout << "Solving with Eigen...\n";
    solveWithEigenLU(A);

    return 0;
}
