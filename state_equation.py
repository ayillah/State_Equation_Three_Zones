import numpy as np
import numpy.linalg as la
from scipy.special import jn, yn
from PDEModel1DWithExactSoln import PDEModel1DWithExactSoln

class StateEquation(PDEModel1DWithExactSoln):
    """
    Exact solution of variable speed problem. Boundary conditions are inflow
    Q_{in}(t) = cos(t) and no refelction at x = L.
    """

    def __init__(self):
        super().__init__(1)
        """Initialize constants."""
        self.c0 = 1.0
        self.alpha = 0.25
        self.a = 0.1
        self.b = 0.9
        self.L = 1.0

    # ----------------------------------------------------------------------
    # Methods of PDEModel1D 

    def speed(self, x):
      """Compute the speed at a specified point x"""
      if x < self.a:
          return self.c0
      elif self.a <= x and x < self.b:
          return self.c0 * np.exp(self.alpha * self.a)
      else:
          return self.c0 * np.exp(self.alpha * self.b)
      
    def f(self, x):

        if x < self.a:
            return x / self.c0
        elif self.a <= x and x < self.b:
            return self.a / self.c0 + (x - self.a) / (self.c0 * np.exp(self.alpha * self.a))
        else:
            return self.a / self.c0 + (self.b - self.a) / (self.c0 * np.exp(self.alpha * self.a)) + (x - self.b) / (self.c0 * np.exp(self.alpha * self.b))

    def applyLeftBC(self, x, t, dx, dt, u):
        """Inflow at the left boundary"""

        left = np.cos(8 * np.pi * t)

        return left
    

    def applyRightBC(self, x, t, dx, dt, u):
        """Setup the right boundary conditons"""

        # Get u at the final and penultimate point
        u_pen = u[:, -2]
        u_ult = u[:, -1]

        # Advect w2 right to left
        right = u_ult - self.speed(x) * (dt / dx) * (u_ult - u_pen)

        return right


    # --------------------------------------------------------------------
    # Method for PDEModel1DWithExactSoln
    def exact_solution(self, X, t):
        """
        Exact solution u(x, t) of our PDE 
        u_t + c(x) * u_x = 0
        
        """
        U = np.zeros_like(X)

        for ix, x in enumerate(X):

            U[ix] = np.cos(8 * np.pi * (t - self.f(x)))
        
        return U