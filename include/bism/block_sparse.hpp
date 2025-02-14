#pragma once

#include <Eigen/Sparse>
#include <vector>
#include <iostream>

class BlockSparseMatrix {
public:
    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Triplet = Eigen::Triplet<double>;

    BlockSparseMatrix(int blocks, int blockSize)
        : numBlocks(blocks), blockSize(blockSize), n(blocks * blockSize) {}

    void generateMatrix() {
        std::vector<Triplet> triplets;
        for (int b = 0; b < numBlocks; ++b) {
            int base = b * blockSize;
            for (int i = 0; i < blockSize; ++i) {
                for (int j = 0; j < blockSize; ++j) {
                    if (i == j || std::abs(i - j) == 1) {
                        triplets.emplace_back(base + i, base + j, (i == j ? 4.0 : -1.0));
                    }
                }
            }
        }
        matrix = SparseMatrix(n, n);
        matrix.setFromTriplets(triplets.begin(), triplets.end());
    }

    const SparseMatrix& getMatrix() const { return matrix; }

private:
    int numBlocks, blockSize, n;
    SparseMatrix matrix;
};
