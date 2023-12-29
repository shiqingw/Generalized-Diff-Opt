#include <cppad/ipopt/solve.hpp>
#include <vector>

namespace {
    using CppAD::AD;

    class FG_eval {
    public:
        typedef std::vector< AD<double> > ADvector;

        void operator()(ADvector& fg, const ADvector& x) {
            assert(fg.size() == 1);
            assert(x.size() == 1);

            // f(x)
            fg[0] = x[0] * x[0] + 3.0;

            // g(x) : constraints
            fg.push_back(x[0] + 1.0);
        }
    };
}

int main() {
    typedef std::vector<double> Dvector;

    // Number of variables (x)
    size_t n = 1;
    // Number of constraints (g)
    size_t m = 1;

    Dvector xi(n), xl(n), xu(n);
    xi[0] = 1.5; // Initial guess
    xl[0] = -4.0; // Lower bound
    xu[0] = 4.0; // Upper bound

    Dvector gl(m), gu(m);
    gl[0] = 0.0; // Lower bound for constraint
    gu[0] = 1.0e19; // Upper bound for constraint

    // Object to compute function and constraints
    FG_eval fg_eval;

    // Options for IPOPT
    std::string options;
    options += "Integer print_level 5\n";
    options += "String sb yes\n";

    // Place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // Solve the problem
    CppAD::ipopt::solve<Dvector, FG_eval>(
        options, xi, xl, xu, gl, gu, fg_eval, solution
    );

    // Output the solution
    std::cout << "x = " << solution.x[0] << std::endl;

    return 0;
}
