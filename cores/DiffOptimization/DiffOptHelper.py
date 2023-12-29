import cvxpy as cp
import numpy as np
import sympy
from sympy import lambdify, Matrix, hessian, diff, Function, zeros
import warnings
import functools

def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} is deprecated", DeprecationWarning)
        return func(*args, **kwargs)
    return wrapper

class DiffOptHelper():
    def __init__(self, cvxpy_prob, constraints_sympy, p_vars, theta_vars):
        """
        Initialize the DiffOptHelper.\n
        Inputs:
            cvxpy_prob: a cvxpy problem
            constraints_sympy: a list of sympy expressions corresponding to the constraints of cvxpy_prob.
                            They should be written using primal_vars and opt_params.
            p_vars: a list of sympy symbols corresponding to the p variables of cvxpy_prob
                        !!DO NOT include the alpha variable!!
            theta_vars: a list of sympy symbols corresponding to the external parameters of cvxpy_prob
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
        self.implicit_dual_vars = [Function("lambda_" + str(i))(*self.theta_vars) for i in range(len(self.constraints_sympy))]
        self.implicit_p_vars = [Function("p_" + str(i))(*self.theta_vars) for i in range(len(self.p_vars))]
        self.build_constraints_dict()
        self.build_matrices()
        self.build_alpha_derivative()

    def build_matrices(self):
        """
        Build the matrices N and b that are used to solve for the gradient of the primal and dual variables.\n
            N: a matrix of shape (dim(p_vars) + dim(constraints_sympy) - 1, dim(p_vars) + dim(constraints_sympy) - 1)\n
            b: a matrix of shape (dim(p_vars) + dim(constraints_sympy) - 1, dim(theta_vars))\n
            N_dtheta: a list of matrices of shape (dim(p_vars) + dim(constraints_sympy) - 1, dim(p_vars) + dim(constraints_sympy) - 1)\n
            b_dtheta: a list of matrices of shape (dim(p_vars) + dim(constraints_sympy) - 1, dim(theta_vars))\n
        """
        N = zeros(len(self.p_vars)+len(self.constraints_sympy)-1, len(self.p_vars)+len(self.constraints_sympy)-1)
        b = zeros(len(self.p_vars)+len(self.constraints_sympy)-1, len(self.theta_vars))
        Q = zeros(len(self.p_vars), len(self.p_vars))
        C = zeros(len(self.constraints_sympy)-1, len(self.p_vars))
        Phi = zeros(len(self.p_vars), len(self.theta_vars))
        Omega = zeros(len(self.constraints_sympy)-1, len(self.theta_vars))
        Q += self.implicit_dual_vars[0] * hessian(self.constraints_sympy[0], self.p_vars)
        Phi += - self.implicit_dual_vars[0] * Matrix([self.constraints_sympy[0]]).jacobian(self.p_vars).jacobian(self.theta_vars)
        cons_A_dp = Matrix([self.constraints_sympy[0]]).jacobian(self.p_vars)
        cons_A_dtheta = Matrix([self.constraints_sympy[0]]).jacobian(self.theta_vars)
        for i in range(1, len(self.constraints_sympy)):
            Q += self.implicit_dual_vars[i] * hessian(self.constraints_sympy[i], self.p_vars)
            Phi += - self.implicit_dual_vars[i] * Matrix([self.constraints_sympy[i]]).jacobian(self.p_vars).jacobian(self.theta_vars)
            C[i-1,:] = Matrix([self.constraints_sympy[i]]).jacobian(self.p_vars) - cons_A_dp
            Omega[i-1,:] = cons_A_dtheta - Matrix([self.constraints_sympy[i]]).jacobian(self.theta_vars)
        N[0:len(self.p_vars),0:len(self.p_vars)] = Q
        N[len(self.p_vars):, 0:len(self.p_vars)] = C
        N[0:len(self.p_vars), len(self.p_vars):] = C.T
        b[0:len(self.p_vars),:] = Phi
        b[len(self.p_vars):,:] = Omega
        substitution_pairs = [[a, b] for a, b in zip(self.p_vars, self.implicit_p_vars)]
        implicit_N = N.subs(substitution_pairs)
        implicit_b = b.subs(substitution_pairs)
        implicit_N_dtheta = [diff(implicit_N, theta_var) for theta_var in self.theta_vars]
        implicit_b_dtheta = [diff(implicit_b, theta_var) for theta_var in self.theta_vars]
        self.N_func = lambdify([self.implicit_p_vars, self.theta_vars, self.implicit_dual_vars], implicit_N, "numpy")
        self.b_func = lambdify([self.implicit_p_vars, self.theta_vars, self.implicit_dual_vars], implicit_b, "numpy")
        self.N_dtheta_func = lambdify([self.implicit_p_vars,
                                       self.theta_vars,
                                       self.implicit_dual_vars,
                                       Matrix(self.implicit_p_vars).jacobian(self.theta_vars),
                                       Matrix(self.implicit_dual_vars).jacobian(self.theta_vars)],
                                       implicit_N_dtheta, "numpy")
        self.b_dtheta_func = lambdify([self.implicit_p_vars, 
                                       self.theta_vars, 
                                       self.implicit_dual_vars,
                                       Matrix(self.implicit_p_vars).jacobian(self.theta_vars),
                                       Matrix(self.implicit_dual_vars).jacobian(self.theta_vars)],
                                       implicit_b_dtheta, "numpy")
    
    def build_alpha_derivative(self):
        """
        Build the function that calculates the derivative of alpha with respect to theta.\n
            alpha: a scalar.\n
            alpha_dtheta: a vector of shape (dim(theta_vars),)\n
            alpha_dthetadtheta: a matrix of shape (dim(theta_vars), dim(theta_vars))\n
        """
        substitution_pairs = [[a, b] for a, b in zip(self.p_vars, self.implicit_p_vars)]
        implicit_alpha = self.constraints_sympy[0].subs(substitution_pairs)
        implicit_alpha_dtheta = Matrix([implicit_alpha]).jacobian(self.theta_vars)
        implicit_alpha_dthetadtheta = hessian(implicit_alpha, self.theta_vars)
        self.alpha_func = lambdify([self.implicit_p_vars, self.theta_vars], implicit_alpha, "numpy")
        self.alpha_dtheta_func = lambdify([self.implicit_p_vars, 
                                           self.theta_vars,
                                           Matrix(self.implicit_p_vars).jacobian(self.theta_vars)],
                                           implicit_alpha_dtheta, "numpy")
        implicit_p_dthetadtheta = [hessian(self.implicit_p_vars[i], self.theta_vars) for i in range(len(self.implicit_p_vars))]
        self.alpha_dthetadtheta_func = lambdify([self.implicit_p_vars, 
                                                 self.theta_vars,
                                                 Matrix(self.implicit_p_vars).jacobian(self.theta_vars),
                                                 *implicit_p_dthetadtheta], implicit_alpha_dthetadtheta, "numpy")

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
            tmp_dict["dthetadp"] = lambdify([self.p_vars, self.theta_vars], Matrix([self.constraints_sympy[i]]).jacobian(self.theta_vars).jacobian(self.p_vars), 'numpy')
            tmp_dict["dthetadtheta"] = lambdify([self.p_vars, self.theta_vars], Matrix([self.constraints_sympy[i]]).jacobian(self.theta_vars).jacobian(self.theta_vars), 'numpy')
            constraints_dict[i] = tmp_dict
        self.constraints_dict = constraints_dict
    
    def get_gradient(self, alpha_val, p_val, theta_val, dual_val):
        """
        Get the gradient of the primal and dual variables with respect to theta.\n
        Inputs:
            alpha_val: a scalar
            p_val: a numpy array of shape (dim(p_vars),)
            theta_val: a numpy array of shape (dim(theta_vars),)
            dual_val: a numpy array of shape (dim(dual_vars),)
        Outputs:
            grad_alpha: a numpy array of shape (1, dim(theta_vars))
            grad_p: a numpy array of shape (dim(p_vars), dim(theta_vars))
            grad_dual: a numpy array of shape (dim(dual_vars), dim(theta_vars))
        """
        threshhold = 1e-4
        active_set_B = []
        for i in range(1, len(self.constraints_sympy)):
            if np.abs(self.constraints_dict[i]["value"](p_val, theta_val) - alpha_val) <= threshhold:
                active_set_B.append(i)
        keep_inds = list(range(len(self.p_vars))) + [i + len(self.p_vars)-1 for i in active_set_B]
        keep_inds = np.array(keep_inds)
        N_total = self.N_func(p_val, theta_val, dual_val)
        b_total = self.b_func(p_val, theta_val, dual_val)
        N = N_total[keep_inds,:][:,keep_inds]
        b = b_total[keep_inds,:]
        X = np.linalg.pinv(N) @ b
        grad_p = X[0:len(p_val), :]
        grad_dual = np.zeros((len(dual_val), len(theta_val)))
        grad_dual[active_set_B,:] = X[len(p_val):len(p_val)+len(active_set_B), :]
        grad_dual[0,:] = - np.sum(grad_dual, axis=0)
        grad_alpha = self.alpha_dtheta_func(p_val, theta_val, grad_p.flatten())
        return grad_alpha, grad_p, grad_dual
    
    def get_gradient_and_hessian(self, alpha_val, p_val, theta_val, dual_val):
        """
        Get the gradient of the primal and dual variables with respect to theta.\n
        Inputs:
            alpha_val: a scalar
            p_val: a numpy array of shape (dim(p_vars),)
            theta_val: a numpy array of shape (dim(theta_vars),)
            dual_val: a numpy array of shape (dim(dual_vars),)
        Outputs:
            grad_alpha: a numpy array of shape (1, dim(theta_vars))
            grad_p: a numpy array of shape (dim(p_vars), dim(theta_vars))
            grad_dual: a numpy array of shape (dim(dual_vars), dim(theta_vars))
            hessian_alpha: a numpy array of shape (dim(theta_vars), dim(theta_vars))
            hessian_p: a list of numpy arrays of shape (dim(p_vars), dim(theta_vars), dim(theta_vars))
            hessian_dual: a list of numpy arrays of shape (dim(dual_vars), dim(theta_vars), dim(theta_vars))
        """
        threshhold = 1e-4
        active_set_B = []
        for i in range(1, len(self.constraints_sympy)):
            if np.abs(self.constraints_dict[i]["value"](p_val, theta_val) - alpha_val) <= threshhold:
                active_set_B.append(i)
        keep_inds = list(range(len(self.p_vars))) + [i + len(self.p_vars)-1 for i in active_set_B]
        keep_inds = np.array(keep_inds)
        N_total = self.N_func(p_val, theta_val, dual_val)
        b_total = self.b_func(p_val, theta_val, dual_val)
        N = N_total[keep_inds,:][:,keep_inds]
        b = b_total[keep_inds,:]
        pinv_N = np.linalg.pinv(N)

        # Calculate the gradient
        X = pinv_N @ b
        grad_p = X[0:len(p_val), :]
        grad_dual = np.zeros((len(dual_val), len(theta_val)))
        grad_dual[active_set_B,:] = X[len(p_val):len(p_val)+len(active_set_B), :]
        grad_dual[0,:] = - np.sum(grad_dual, axis=0)
        grad_alpha = self.alpha_dtheta_func(p_val, theta_val, grad_p.flatten())

        # Calculate the hessian
        X_dtheta_total = np.zeros((X.shape[0], X.shape[1], len(theta_val)))
        N_dtheta_total = self.N_dtheta_func(p_val, theta_val, dual_val, grad_p.flatten(), grad_dual.flatten())
        b_dtheta_total = self.b_dtheta_func(p_val, theta_val, dual_val, grad_p.flatten(), grad_dual.flatten())
        for i in range(len(theta_val)):
            N_dtheta = N_dtheta_total[i][keep_inds,:][:,keep_inds]
            b_dtheta = b_dtheta_total[i][keep_inds,:]
            X_dtheta = - pinv_N @ N_dtheta @ pinv_N @ b + pinv_N @ b_dtheta
            X_dtheta_total[:,:,i] = X_dtheta
        hessian_p = X_dtheta_total[0:len(p_val),:,:]
        hessian_dual = np.zeros((len(dual_val), len(theta_val), len(theta_val)))
        hessian_dual[active_set_B,:,:] = X_dtheta_total[len(p_val):len(p_val)+len(active_set_B),:,:]
        hessian_dual[0,:,:] = - np.sum(hessian_dual, axis=0)
        hessian_p_val = [hess.flatten() for hess in hessian_p]
        heissian_alpha = self.alpha_dthetadtheta_func(p_val, theta_val, grad_p.flatten(), *hessian_p_val)
        return grad_alpha, grad_p, grad_dual, heissian_alpha, hessian_p, hessian_dual

    @ deprecated
    def get_gradient_old(self, alpha_val, p_val, theta_val, dual_val):
        """
        Get the gradient of the primal and dual variables with respect to theta.\n
        Inputs:
            alpha_val: a scalar
            p_val: a numpy array of shape (dim(p_vars),)
            theta_val: a numpy array of shape (dim(theta_vars),)
            dual_val: a numpy array of shape (dim(dual_vars),)
        Outputs:
            grad_alpha: a numpy array of shape (1, dim(theta_vars))
            grad_p: a numpy array of shape (dim(p_vars), dim(theta_vars))
            grad_dual: a numpy array of shape (dim(dual_vars), dim(theta_vars))
        """
        threshhold = 1e-4
        lambda_A = dual_val[0]
        active_set_B = []
        for i in range(1, len(self.constraints_sympy)):
            if np.abs(self.constraints_dict[i]["value"](p_val, theta_val) - alpha_val) <= threshhold:
                active_set_B.append(i)
        active_set_B = np.array(active_set_B)
        N = np.zeros((len(p_val)+len(active_set_B), len(p_val)+len(active_set_B)))
        b = np.zeros((len(p_val)+len(active_set_B), len(theta_val)))
        C = np.zeros((len(active_set_B), len(p_val)))
        Omega = np.zeros((len(active_set_B), len(theta_val)))
        Q = lambda_A * self.constraints_dict[0]["dpdp"](p_val, theta_val)
        Phi = - lambda_A * self.constraints_dict[0]["dpdtheta"](p_val, theta_val)
        cons_A_dp = self.constraints_dict[0]["dp"](p_val, theta_val)
        cons_A_dtheta = self.constraints_dict[0]["dtheta"](p_val, theta_val)
        for i in range(len(active_set_B)):
            num = active_set_B[i]
            Q += dual_val[num] * self.constraints_dict[num]["dpdp"](p_val, theta_val)
            Phi += - dual_val[num] * self.constraints_dict[num]["dpdtheta"](p_val, theta_val)
            C[i,:] = self.constraints_dict[num]["dp"](p_val, theta_val) - cons_A_dp
            Omega[i,:] = cons_A_dtheta - self.constraints_dict[num]["dtheta"](p_val, theta_val)
        N[0:len(p_val), 0:len(p_val)] = Q
        N[len(p_val):len(p_val)+len(active_set_B), 0:len(p_val)] = C
        N[0:len(p_val), len(p_val):len(p_val)+len(active_set_B)] = C.T
        b[0:len(p_val), :] = Phi
        b[len(p_val):len(p_val)+len(active_set_B), :] = Omega
        X = np.linalg.pinv(N) @ b
        grad_p = X[0:len(p_val), :]
        grad_dual = np.zeros((len(dual_val), len(theta_val)))
        grad_dual[active_set_B,:] = X[len(p_val):len(p_val)+len(active_set_B), :]
        grad_dual[0,:] = - np.sum(grad_dual, axis=0)
        grad_alpha = cons_A_dp @ grad_p + cons_A_dtheta
        return grad_alpha, grad_p, grad_dual
        