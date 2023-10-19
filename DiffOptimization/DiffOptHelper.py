import cvxpy as cp
import numpy as np
import sympy
from sympy import lambdify, Matrix, hessian

class DiffOptHelper():
    def __init__(self, cvxpy_prob, constraints_sympy, primal_vars, dual_vars, opt_params):
        """
        Initialize the DiffOptHelper
        Inputs:
            cvxpy_prob: a cvxpy problem
            primal_vars: a list of sympy symbols corresponding to the primal variables of cvxpy_prob
                        DO NOT include the alpha variable
            dual_vars: a list of sympy symbols corresponding to the dual variables of cvxpy_prob
            opt_params: a list of sympy symbols corresponding to the external parameters of cvxpy_prob
            constraints_sympy: a list of sympy expressions corresponding to the constraints of cvxpy_prob.
                            They should be written using primal_vars and opt_params.
        """
        if not isinstance(cvxpy_prob, cp.Problem):
            raise TypeError("cvxpy_prob must be a cvxpy problem")
        if not isinstance(constraints_sympy, list) or not isinstance(constraints_sympy[0], sympy.Expr):
            raise TypeError("constraints_sympy must be a list of sympy expressions \
                            corresponding to the constraints of cvxpy_prob")
        self.problem = cvxpy_prob
        self.constraints_sympy = constraints_sympy
        self.primal_vars = primal_vars
        self.dual_vars = dual_vars
        self.opt_params = opt_params


    def get_gradient(self):
        pass

    def get_hessian(self):
        pass