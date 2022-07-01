#include "matrix.h"

#include <gtest/gtest.h>

TEST(MatrixTests, testSolve) { // A * x = B
    Matrix A({{10, 6, 2,  0},
              {5,  1, -2, 4},
              {3,  5, 1,  -1},
              {0,  6, -2, 2}});
    Matrix B({{25},
              {14},
              {10},
              {8}});
    Matrix expectedX({{2},
                      {1},
                      {-0.5},
                      {0.5}});
    Matrix x = Solve(A, B);

    ASSERT_EQ(x, expectedX);
}

TEST(MatrixTests, testInverse2) { // Inverse matrix
    Matrix A({{0, 0, 0, 1},
              {0, 0, 2, 0},
              {0, 3, 0, 0},
              {4, 0, 0, 0}});
    Matrix expectedInv({{0.25, 0,   0,     0},
                        {0, 1.0/3,  0, 0},
                        {0, 0, 0.5,     0},
                        {0, 0,   0,     1}});
    A.Inverse();
    for (size_t i = 0; i < A.get_rows(); ++i) {
        for (size_t j = 0; j < A.get_rows(); ++j) {
            std::cout << A.get_mat()[i][j] << " ";
        }
        std::cout << std::endl;
    }
    ASSERT_EQ(Matrix(A.get_mat()), expectedInv);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}