#include <bism/generate_eigen_matrices.hpp>

std::pair<DenseRowMajor, DenseColMajor> create_dense_matrices(int num_species, int num_cells) {
    DenseRowMajor species_major(num_species, num_cells);
    for(int i = 0; i < num_species; i++) {
        for(int j = 0; j < num_cells; j++) {
            species_major(i, j) = 1.0 * (i + 1) + 0.1 * (j + 1);
        }
    }

    DenseColMajor cell_major(num_species, num_cells);
    for(int i = 0; i < num_cells; i++) {
        for(int j = 0; j < num_species; j++) {
            cell_major(j, i) = 1.0 * (j + 1) + 0.1 * (i + 1);
        }
    }

    return {species_major, cell_major};
}

std::pair<SparseRowMajor, SparseColMajor> create_sparse_matrices(int num_species, int num_cells) {
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

    return {species_major, cell_major};
}