import cvxpy as cp
import numpy as np
import sympy
from sympy import lambdify, Matrix, hessian, diff, Function

class DiffOptHelper():
    def __init__(self, cvxpy_prob, constraints_sympy, p_vars, theta_vars):
        """
        Initialize the DiffOptHelper.\n
        Inputs:
            cvxpy_prob: a cvxpy problem
            constraints_sympy: a list of sympy expressions corresponding to the constraints of cvxpy_prob.
                            They should be written using primal_vars and opt_params.
            p_vars: a list of sympy symbols corresponding to the p variables of cvxpy_prob
                        DO NOT include the alpha variable
            theta_vars: a list of sympy symbols corresponding to the external parameters of cvxpy_prob
            dual_vars: a list of sympy symbols corresponding to the dual variables of cvxpy_prob
        """
        if not isinstance(cvxpy_prob, cp.Problem):
            raise TypeError("cvxpy_prob must be a cvxpy problem")
        if not isinstance(constraints_sympy, list) or not isinstance(constraints_sympy[0], sympy.Expr):
            raise TypeError("constraints_sympy must be a list of sympy expressions \
                            corresponding to the constraints of cvxpy_prob")
        self.problem = cvxpy_prob
        self.constraints_sympy = constraints_sympy
        self.p_vars = p_vars
        self.theta_vars = theta_vars
        self.dual_vars = [Function("lambda_" + str(i))(*self.theta_vars) for i in range(len(constraints_sympy))]
        self.build_constraints_dict()

    def build_constraints_dict(self):
        """
        Build a dictionary that stores the lambdified constraints and derivatives of the constraints.\n
        The value constraints_dict[i] is a dictionary that stores:\n
            value: dim(p_vars), dim(theta_vars) -> a scalar
            dp: dim(p_vars), dim(theta_vars) -> dim(p_vars)
            dpdp: dim(p_vars), dim(theta_vars) -> dim(p_vars) x dim(p_vars)
            dtetha: dim(p_vars), dim(theta_vars) -> dim(theta_vars)
            dpdtheta: dim(p_vars), dim(theta_vars) -> dim(p_vars) x dim(theta_vars)

        """
        constraints_dict = {}
        for i in range(len(self.constraints_sympy)):
            tmp_dict = {}
            tmp_dict["value"] = lambdify([self.p_vars, self.theta_vars], self.constraints_sympy[i], 'numpy')
            tmp_dict["dp"] = lambdify([self.p_vars, self.theta_vars], Matrix([self.constraints_sympy[i]]).jacobian(self.p_vars), 'numpy')
            tmp_dict["dpdp"] = lambdify([self.p_vars, self.theta_vars], hessian(self.constraints_sympy[i], self.p_vars), 'numpy')
            tmp_dict["dtheta"] = lambdify([self.p_vars, self.theta_vars], Matrix([self.constraints_sympy[i]]).jacobian(self.theta_vars), 'numpy')
            tmp_dict["dpdtheta"] = lambdify([self.p_vars, self.theta_vars], Matrix([self.constraints_sympy[i]]).jacobian(self.p_vars).jacobian(self.theta_vars), 'numpy')
            constraints_dict[i] = tmp_dict
        self.constraints_dict = constraints_dict

    def get_gradient(self, p_val, theta_val, dual_val):
        """
        Get the gradient of the primal and dual variables with respect to theta.\n
        Inputs:
            p_val: a numpy array of shape (dim(p_vars),)
            theta_val: a numpy array of shape (dim(theta_vars),)
            dual_val: a numpy array of shape (dim(dual_vars),)
        Outputs:
            grad_p: a numpy array of shape (dim(p_vars), dim(theta_vars))
            grad_alpha: a numpy array of shape (1, dim(theta_vars))
            grad_dual: a numpy array of shape (dim(dual_vars), dim(theta_vars))
        """

        pass

    def get_hessian(self):
        pass