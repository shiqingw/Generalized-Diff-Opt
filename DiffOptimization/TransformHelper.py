import numpy as np
import sympy
from sympy import lambdify, Matrix, hessian

class TransformHelper():
    def __init__(self, states, transformed_states):
        """
        Initialize the differentiable optimization problem
        Inputs: 
            states: list of sympy symbols
            transformed_states: list of sympy expressions that should be written using states
        """
        if not isinstance(states, list) or not isinstance(states[0], sympy.Symbol):
            raise TypeError("states must be a list of sympy symbols")
        if not isinstance(transformed_states, list) or not isinstance(transformed_states[0], sympy.Expr):
            raise TypeError("params must be a list of sympy expressions")
        self.states = states
        self.transformed_states = transformed_states
        self.transform_func = lambdify(self.states, self.transformed_states, 'numpy')
        self.jacobian_func = lambdify(self.states, Matrix(self.transformed_states).jacobian(self.states), 'numpy')
        hessians = [hessian(f_i, self.states) for f_i in self.transformed_states]
        self.hessian_func = lambdify(self.states, hessians, 'numpy')

    def get_transformed_states_from_states(self, states):
        """
        Get transformed states from states
        Inputs:
            states: a numpy array
        Outputs:
            transformed_states: a numpy array
        """
        return np.array(self.transform_func(*states))

    def get_transformation_jacobian(self, states):
        """
        Get the jacobian of the transformation function
        Inputs:
            states: a numpy array
        Outputs:
            jacobian: a numpy array of size dim(transformed_states) x dim(states)
        """
        return np.array(self.jacobian_func(*states))

    def get_transformation_hessian(self, states):
        """
        Get the hessian of the transformation function
        Inputs:
            states: a numpy array
        Outputs:
            hessian: a numpy array of size dim(transformed_states) x dim(states) x dim(states)
        """
        return np.array(self.hessian_func(*states))
    
    def get_backward_jacobian(self, states, external_jacobian):
        """
        Get the jacobian w.r.t the original states form the jacobian w.r.t the transformed states.
        This is done by chain rule:
            J_original = J_external * J_transform
        Inputs:
            states: a numpy array
            external_jacobian: a numpy array of size external_jacobian.shape[0] x dim(transformed_states)
        Outputs:
            jacobian: a numpy array of size external_jacobian.shape[0] x dim(states)
        """
        if not isinstance(external_jacobian, np.ndarray):
            raise TypeError("external_jacobian must be a numpy array")
        J_transform = self.get_transformation_jacobian(states)
        return external_jacobian @ J_transform

    def get_backward_hessian(self, states, external_jacobian, external_hessian):
        """
        Get the hessian w.r.t the original states form the hessian w.r.t the transformed states.
        This is done by chain rule:
            H_original = J_transform^T * H_external * J_transform + J_external * H_transform
        Inputs:
            states: a numpy array
            external_jacobian: a numpy array of size external_jacobian.shape[0] x dim(transformed_states)
            external_hessian: a numpy array of size external_hessian.shape[0] x dim(transformed_states) x dim(transformed_states)
        Outputs:
            hessian: a numpy array of size external_hessian.shape[0] x dim(states) x dim(states)
        """
        if not isinstance(external_jacobian, np.ndarray):
            raise TypeError("external_jacobian must be a numpy array")
        if not isinstance(external_hessian, np.ndarray):
            raise TypeError("external_hessian must be a numpy array")
        while external_jacobian.ndim < 2:
            external_jacobian = external_jacobian[np.newaxis, ...]
        while external_hessian.ndim < 3:
            external_hessian = external_hessian[np.newaxis, ...]
        J_transform = self.get_transformation_jacobian(states)
        H_transform = self.get_transformation_hessian(states)
        term1 = np.einsum('lj,ijk->ilk', J_transform.T, external_hessian)
        term1 = np.einsum('ijk,kl->ijl', term1, J_transform)
        term2 = np.einsum('ij,jkl->ikl', external_jacobian, H_transform)
        return np.squeeze(term1 + term2)
