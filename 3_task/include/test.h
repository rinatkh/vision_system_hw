#pragma once

#include <Eigen/Core>

// variant 9
const double a = 0.3;
const double b = 2.1;
const double d = 0.5;

Eigen::VectorXd my_sin(double time, const Eigen::VectorXd &val);

[[maybe_unused]] Eigen::VectorXd simple1(double time, const Eigen::VectorXd &val);

[[maybe_unused]] Eigen::VectorXd simple2(double time, const Eigen::VectorXd &val);

[[maybe_unused]] Eigen::VectorXd result1(double time);

[[maybe_unused]] Eigen::VectorXd result2(double time);

void test1();

void test2();
