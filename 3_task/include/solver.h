#pragma once
#include <functional>
#include <vector>
#include <memory>

#include <Eigen/Core>

class Solver;

class Strategy {
public:
    explicit Strategy(Solver &solver, double step = 0.1) : solver_(solver), step_(step) {}

    virtual void calc_step() = 0;

    Eigen::MatrixXd gen_tmp();

    [[nodiscard]] double get_step() const { return step_; }
    void set_step(double step) { step_ = step; }

protected:
    Solver &solver_;
    double step_;
};

class StrategyDP : public Strategy {
public:
    StrategyDP(double max_diff, double min_diff,
               Solver &solver, double step = 0.1) : Strategy(solver, step),
                                                  max_diff_(max_diff),
                                                  min_diff_(min_diff) {}

    void calc_step() override;

private:
    double max_diff_;
    double min_diff_;
};

class StrategyRunge : public Strategy {
public:
    using Strategy::Strategy;

    void calc_step() override;
};


class Solver {
public:
    friend class Walker;
    friend class DPWalker;
    friend class RungeWalker;

    Solver(const Eigen::MatrixXd &butcher_matrix,
           std::function<Eigen::VectorXd(double, const Eigen::VectorXd &)> recalc_function,
           double max_diff = 10e-10, double min_diff = 10e-10);

    void init_values(const Eigen::VectorXd &init_vals, double init_time);

    void calc_step() const { walker_->calc_step(); }
    [[nodiscard]] double t() const { return time_; }
    [[nodiscard]] Eigen::VectorXd& vals() { return values_; }
    [[nodiscard]] double get_step() const { return walker_->get_step(); }

    void set_step(double step) const { walker_->set_step(step); }

public:
    std::unique_ptr<Strategy> walker_;

    std::function<Eigen::VectorXd(double, const Eigen::VectorXd &)> recalc_function_;
    Eigen::VectorXd values_;
    Eigen::VectorXd steps_;

    Eigen::MatrixXd a_;
    std::vector<Eigen::MatrixXd> b_;

    double time_ = 0;
};

Eigen::MatrixXd get_runge();
Eigen::MatrixXd get_DP();