#include "matrix.h"

static const char *BAD_INPUT_PARAMETERS = "Bad Input Parameters";
static const char *BAD_SIZE_PARAMETERS = "Bad Size Parameters";
static const char *BAD_MATRIX_SIZES = "not correct sizes of matrix";
static const double EPS = 1e-7;

Matrix::Matrix(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) throw std::runtime_error(BAD_INPUT_PARAMETERS);
    data.resize(rows);
    for (auto& raw : data) {
        raw.resize(cols);
    }
}

Matrix::Matrix(const std::vector<std::vector<double>> &mat) {
    if (mat.empty() || mat.data()->empty()) throw std::runtime_error(BAD_INPUT_PARAMETERS);
    data = mat;
}

double Matrix::operator()(size_t i, size_t j) const {
    if (i >= get_rows() || j >= get_cols()) throw std::runtime_error(BAD_SIZE_PARAMETERS);
    return data[i][j];
}

double &Matrix::operator()(size_t i, size_t j) {
    if (i >= get_rows() || j >= get_cols()) throw std::runtime_error(BAD_SIZE_PARAMETERS);
    return data[i][j];
}
bool Matrix::operator==(const Matrix &mat) const {
    auto rows_ = get_rows();
    auto cols_ = get_cols();
    if (rows_ != mat.get_rows() || cols_ != mat.get_cols())
        throw std::runtime_error(BAD_MATRIX_SIZES);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (std::abs(mat(i, j) - (*this)(i, j)) > EPS) {
                return false;
            }
        }
    }
    return true;
}


Matrix Matrix::operator*(double val) const {
    Matrix result(data);

    for (auto &one_row: result.data) {
        for (auto &one_val: one_row) {
            one_val *= val;
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix &mat) const {
    auto rows_ = get_rows();
    auto cols_ = get_cols();

    auto val_rows_ = mat.get_rows();
    auto val_cols_ = mat.get_cols();

    if (cols_ != val_rows_) throw std::invalid_argument(BAD_MATRIX_SIZES);

    Matrix result(rows_, val_cols_);

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < val_cols_; ++j) {
            for (size_t k = 0; k < cols_; ++k) {
                result(i, j) += data[i][k] * mat(k, j);
            }
        }
    }

    return result;
}

Matrix Matrix::operator+(const Matrix &mat) const {
    auto rows_ = get_rows();
    auto cols_ = get_cols();
    if (rows_ != mat.get_rows() || cols_ != mat.get_cols())
        throw std::runtime_error(BAD_MATRIX_SIZES);

    Matrix result(data);

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(i, j) += mat(i, j);
        }
    }

    return result;
}

Matrix Matrix::operator-(const Matrix &mat) const {
    auto rows_ = get_rows();
    auto cols_ = get_cols();
    if (rows_ != mat.get_rows() || cols_ != mat.get_cols())
        throw std::runtime_error(BAD_MATRIX_SIZES);

    Matrix result(data);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(i, j) -= mat(i, j);
        }
    }

    return result;
}

std::pair<Matrix, Matrix> Matrix::LU() const {
    auto rows = get_rows();
    auto cols = get_cols();
    Matrix L(rows, cols);
    Matrix U(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; j++) {
            if (j < i) {
                U(i, j) = 0;
            } else {
                U(i, j) = (*this)(i, j);
                for (size_t k = 0; k < i; k++) {
                    U(i, j) -= L(i, k) * U(k, j);
                }
            }
        }
        for (size_t j = 0; j < cols; ++j) {
            if (j < i) {
                L(j, i) = 0;
            } else if (j == i) {
                L(i, j) = 1;
            } else {
                L(j, i) = (*this)(j, i) / U(i, i);
                for (size_t k = 0; k < i; ++k) {
                    L(j, i) -= L(j, k) * U(k, i) / U(i, i);
                }
            }
        }
    }
    return {L, U};
}

void Matrix::Inverse() {
    auto rows = get_rows(); auto cols = get_cols();
    Matrix invMat(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        invMat(i, i) = 1;
    }
    auto LU = (*this).LU();
    for (size_t j = 0; j < rows; j++) {
        for (size_t i = 0; i < cols; i++) {
            for (size_t k = 0; k < i; k++) {
                invMat(i, j) -= LU.first(i, k) * invMat(k, j);
            }
            invMat(i, j) /= LU.first(i, i);
        }
        for (int i = rows - 1; i >= 0; i--) {
            for (size_t k = i + 1; k < rows; k++) {
                invMat(i, j) -= LU.second(i, k) * invMat(k, j);
            }

            invMat(i, j) /= LU.second(i, i);
        }
    }
    *this = invMat;
}

double Matrix::det() const {
    if (get_rows() != get_cols() || get_rows() == 0) {
        throw std::invalid_argument(BAD_SIZE_PARAMETERS);
    }

    auto [_, U] = LU();

    double det = data[0][0];
    for (size_t i = 1; i < get_rows(); ++i) {
        det *= U(i, i);
    }

    return det;
}

std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
    size_t cols = matrix.get_cols();
    size_t rows = matrix.get_rows();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            os << matrix(i, j);
            if (j == cols - 1) {
                os << "\n";
            } else {
                os << " ";
            }
        }
    }
    return os;
}

static Matrix forward(const Matrix& L, const Matrix& b) {
    Matrix y(L.get_rows(), 1);
    for (size_t i = 0; i < L.get_rows(); ++i) {
        y(i, 0) = b(i, 0);
        for (size_t j = 0; j < i; ++j) {
            y(i, 0) -= L(i, j) * y(j, 0);
        }
        y(i, 0) /= L(i, i);
    }
    return y;
}

static Matrix back(const Matrix& U, const Matrix& y) {
    Matrix x(U.get_rows(), 1);
    for (int i = U.get_rows() - 1; i >= 0; --i) {
        x(i, 0) = y(i, 0);
        for (size_t j = i + 1; j < U.get_rows(); ++j) {
            x(i, 0) -= U(i, j) * x(j, 0);
        }
        x(i, 0) /= U(i, i);
    }
    return x;
}

Matrix Solve(const Matrix &A, const Matrix &B) {
    auto LU = A.LU();
    Matrix y = forward(LU.first, B);
    Matrix x = back(LU.second, y);
    return x;
}
