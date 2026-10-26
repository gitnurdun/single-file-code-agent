#include <iostream>
#include <vector>
#include <cmath>

// Function to swap two rows in a matrix
void swapRows(std::vector<std::vector<double>>& matrix, int row1, int row2) {
    std::vector<double> temp = matrix[row1];
    matrix[row1] = matrix[row2];
    matrix[row2] = temp;
}

// Function to perform Gaussian elimination to find the inverse of a matrix
std::vector<std::vector<double>> matinv(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    std::vector<std::vector<double>> augmentedMatrix(A);
    std::vector<std::vector<double>> inverse(n, std::vector<double>(n));

    // Perform Gaussian elimination
    for (int i = 0; i < n; ++i) {
        // Find the pivot element
        double maxVal = std::abs(augmentedMatrix[i][i]);
        int pivotRow = i;
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(augmentedMatrix[j][i]) > maxVal) {
                maxVal = std::abs(augmentedMatrix[j][i]);
                pivotRow = j;
            }
        }

        // Swap rows to bring the pivot element to the diagonal
        swapRows(augmentedMatrix, i, pivotRow);
        swapRows(inverse, i, pivotRow);

        // Divide the pivot row by the pivot element
        double pivotElement = augmentedMatrix[i][i];
        for (int j = 0; j < n; ++j) {
            augmentedMatrix[i][j] /= pivotElement;
            inverse[i][j] /= pivotElement;
        }

        // Eliminate the pivot element from the other rows
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                double factor = augmentedMatrix[j][i];
                for (int k = 0; k < n; ++k) {
                    augmentedMatrix[j][k] -= factor * augmentedMatrix[i][k];
                    inverse[j][k] -= factor * inverse[i][k];
                }
            }
        }
    }

    return inverse;
}

int main() {
    std::vector<std::vector<double>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<std::vector<double>> inverse = matinv(A);

    std::cout << "Inverse of A:\n";
    for (const auto& row : inverse) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    return 0;
}