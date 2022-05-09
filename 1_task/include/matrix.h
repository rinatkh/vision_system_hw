#pragma once

#include <vector>
#include <iostream>


class Matrix {
public:
    Matrix() = delete;

    ~Matrix() = default;

    explicit Matrix(size_t rows, size_t cols);

    explicit Matrix(const std::vector<std::vector<double>> &mat);

    Matrix(const Matrix &mat) = default;

    Matrix &operator=(const Matrix &mat) = default;

    [[nodiscard]] size_t get_rows() const { return data.size(); }

    [[nodiscard]] size_t get_cols() const { return data.data()->size(); }

    [[nodiscard]] std::vector<std::vector<double>> get_mat() const { return data; }

    double operator()(size_t i, size_t j) const;

    double &operator()(size_t i, size_t j);

    bool operator==(const Matrix &mat) const;

    bool operator!=(const Matrix &mat) const { return !(*this == mat); }

    Matrix operator*(double val) const;

    Matrix operator*(const Matrix &mat) const;

    Matrix operator+(const Matrix &val) const;

    Matrix operator-(const Matrix &mat) const;

    [[nodiscard]] std::pair<Matrix, Matrix> LU() const;

    void Inverse();

    [[nodiscard]] double det() const;

private:
    std::vector<std::vector<double>> data;

    friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix);
};

Matrix Solve(const Matrix &A, const Matrix &b);
