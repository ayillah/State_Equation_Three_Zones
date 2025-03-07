import numpy as np
import numpy.linalg as la
from scipy.special import jn, yn
from PDEModel1DWithExactSoln import PDEModel1DWithExactSoln

class ThreeZoneVarSpeed(PDEModel1DWithExactSoln):
    """
    Exact solution of variable speed problem. Boundary conditions are inflow
    Q_{in}cos(Omega t) and terminal resistance R.
    """

    def __init__(self):
        super().__init__(2)
        """Initialize constants."""
        self.c0 = 1.0
        self.gamma = 1.25 * np.log(2)
        self.a = 0.1
        self.b = 0.9
        self.L = 1.0
        self.A_r = 0.5
        self.rho = 1.0
        self.R = 1.5
        self.Q_in = 1.0
        self.Omega = 8.0 * np.pi
        self.getCoefficients()

    # ----------------------------------------------------------------------
    # Methods of PDEModel1D 

    def speed(self, x):
      """Compute the speed at a specified point x"""
      if x < self.a:
          return self.c0
      elif self.a <= x and x < self.b:
          return self.c0 * np.exp(self.gamma * (x - self.a) )
      else:
          return self.c0 * np.exp(self.gamma * (self.b-self.a))
      
    def jacobian(self, x):
        """Compute the Jacobian at a point x"""
        c = self.speed(x)

        return np.array([[0, 2 * (c ** 2)], [0.5, 0]])
    
    def jacEigensystem(self, x):
        c = self.speed(x)
        V = np.array([[2 * c, -2 * c], [1, 1]])
        lam = np.array([c,-c])
        return (lam, V)

    def jacEigenvalues(self, x):
        c = self.speed(x)
        lam = np.array([c,-c])

        return lam

    def applyLeftBC(self, x, t, dx, dt, u):
        
        # Get current solution at points indexed 0 and 1
        uAt0 = u[:,0] 
        uAt1 = u[:,1]

        # Get the eigenvalues and eigenvectors 
        lam, V = self.jacEigensystem(x)

        # Solve the equation V*w = u for the Riemann variables at x0 and x1
        wAt0 = la.solve(V, uAt0)
        wAt1 = la.solve(V, uAt1) 

        w2At0 = wAt0[1]
        w2At1 = wAt1[1]
        
        # With inflow BCs, the first Riemann variable is specified. The second
        # Riemann variable is advected backwards
        w1 = self.Q_in * np.cos(self.Omega * t)
        #w1 = 0.5 * self.speed(x) * self.rho * self.Q_in / self.A_r * np.cos(self.Omega * t)
        w2 = w2At0 - lam[1] * (dt/dx) * (w2At1 - w2At0)

        uNextAt0 = np.matmul(V, np.array([w1,w2]))

        return uNextAt0
    

    def applyRightBC(self, x, t, dx, dt, u):
        """Setup the right boundary conditons"""

        c = self.speed(x)
        R = self.R

        # Get u at the final and penultimate point
        u_pen = u[:, -2]
        u_ult = u[:, -1]

        # Get the eigenvalues and eigenvectors 
        lam, V = self.jacEigensystem(x)

        # Solve the equation V*w = u for the Riemann variables at x0 and x1
        w_pen = la.solve(V, u_pen)
        w_ult = la.solve(V, u_ult) 

        w1_pen = w_pen[0]
        w1_ult = w_ult[0]

        # Advect w1 left to right
        w1 = w1_ult - lam[0] * (dt / dx) * (w1_ult - w1_pen)

        # Solve the equation p = R q for w2
        w2 = ((2 * c - R) / (2 * c + R)) * w1
        #print('w2=',w2)

        wNext = np.array([w1, w2])

        uNext = np.matmul(V, wNext)

        return uNext


    # --------------------------------------------------------------------
    # Method for PDEModel1DWithExactSoln
    def exact_solution(self, X, t):
        """
        Exact solution p(x, t) and q(x, t) of our PDE system
        p_t + (rho * c(x)^2)/A_r * q_x = 0
        q_t + (A_r / rho) * p_x = 0
        """
        i = 1.0j

        d2 = np.exp(i * self.Omega * t)
        d3 = (self.A_r) / (self.rho * self.Omega)

        P = np.copy(X)
        Q = np.copy(X)
        for ix, x in enumerate(X):
            if x <= self.a: # zone 1
                (psi1, psi2, dpsi1, dpsi2) = self.zone1Psi(x)
                A = self.C[0]
                B = self.C[1]
            elif x > self.a and x <= self.b:
                (psi1, psi2, dpsi1, dpsi2) = self.zone2Psi(x)
                A = self.C[2]
                B = self.C[3]
            else:
                (psi1, psi2, dpsi1, dpsi2) = self.zone3Psi(x)
                A = self.C[4]
                B = self.C[5]

            P[ix] = 2.0 * np.real((A * psi1 + B * psi2) * d2) 
            Q[ix] = 2.0 * np.real(i * d3 * (A * dpsi1 + B * dpsi2) * d2)

            
        # stack P,Q into U
        U = np.vstack((P, Q))
        
        return U
        


    # ------------------------------------------------------------------------
    # Internal methods specific to this model. The user code should never
    # need to call these.

    def zone1Psi(self, x):
        """Evaluate the basis functions in zone 1"""
        i = 1.0j
        c1 = self.c0

        psi1 = np.exp((i * self.Omega * x) / c1)
        psi2 = np.exp((-i * self.Omega * x)/ c1)
                      
        dpsi1 = (i * self.Omega / c1) * np.exp((i * self.Omega * x) / c1)
        dpsi2 = (-i * self.Omega / c1) * np.exp((-i * self.Omega * x) / c1)

        return (psi1, psi2, dpsi1, dpsi2)

    def zone2Psi(self, x):
        """Evaluate the basis functions in zone 2"""
        zeta = np.exp(-self.gamma * (x - self.a))
        kappa = (self.Omega) / (self.gamma * self.c0)

        psi1 = jn(0, kappa * zeta)
        psi2 = yn(0, kappa * zeta)

        dpsi1 = self.gamma * kappa * zeta * jn(1, kappa * zeta)
        dpsi2 = self.gamma * kappa * zeta * yn(1, kappa * zeta)

        return (psi1, psi2, dpsi1, dpsi2)

    def zone3Psi(self, x):
        """Evaluate the basis functions in zone 2"""
        i = 1.0j
        c1 = self.c0
        c3 = self.c0 * np.exp(self.gamma * self.b - self.gamma * self.a)

        psi1 = np.exp((i * x * self.Omega) / c3)
        psi2 = np.exp((-i * x * self.Omega) / c3)

        dpsi1 = (i * self.Omega * np.exp(-self.gamma * (self.b - self.a) + (i * x * self.Omega) / c3)) / c1
        dpsi2 = (-i * self.Omega * np.exp(-self.gamma * (self.b - self.a) - (i * x * self.Omega) / c3)) / c1

        return (psi1, psi2, dpsi1, dpsi2)
    
    def getCoefficients(self):
        """Set up Wronskians and compute solutions in each zone."""
        i = 1.0j

        # Define the wave speeds

        # Constant speed c1 in zone 1
        c1 = self.c0

        # Constant speed c3 in zone 3
        c3 = self.c0 * np.exp(self.gamma * self.b - self.gamma * self.a)

        # Construct basis elements

        # Zone 1-2 interface
        (psi11, psi12, dpsi11, dpsi12) = self.zone1Psi(self.a)
        (psi21, psi22, dpsi21, dpsi22) = self.zone2Psi(self.a)
        
        # Zone 2-3 interface
        (psi021, psi022, dpsi021, dpsi022) = self.zone2Psi(self.b)
        (psi31, psi32, dpsi31, dpsi32) = self.zone3Psi(self.b)

        # Form Wronskians
        W1 = np.array([[psi11, psi12], [dpsi11, dpsi12]])
        W2 = np.array([[psi21, psi22], [dpsi21, dpsi22]])
        W02 = np.array([[psi021, psi022], [dpsi021, dpsi022]])
        W3 = np.array([[psi31, psi32], [dpsi31, dpsi32]])

        # Boundary Conditions
        # Inflow BC

        # RHS of the inflow BC
        rhs1 = np.complex128((0.5 * self.c0 * self.rho * self.Q_in) / self.A_r)
        rhs2 = np.zeros((2, 1))
        rhs3 = np.zeros((2, 1))

        rhs = np.vstack((rhs1, rhs2, rhs3, 0.0))

        # Interface conditions

        # Create block matrices
        M11 = np.array([0, 1])
        M12 = np.array([0, 0])
        M23 = np.array([[0, 0], [0, 0]])

        # Terminal resistance BC
        sr1 = (np.exp(-self.gamma * (self.b - self.a) + (i * self.L * self.Omega) / c3) * (c3 * self.rho + self.R * self.A_r)) / (c1 * self.rho)
        sr2 = (np.exp(-self.gamma * (self.b - self.a) - (i * self.L * self.Omega) / c3) * (c3 * self.rho - self.R * self.A_r)) / (c1 * self.rho)

        S_R = np.array([sr1, sr2])

        # Construct the matrix M of block matrices
        row1 = np.hstack((M11, M12, M12))
        row2 = np.hstack((W1, -W2, M23))
        row3 = np.hstack((M23, W02, -W3))
        row4 = np.hstack((M12, M12, S_R))

        # Solve for the coefficients vector C
        M = np.vstack((row1, row2, row3, row4))
        self.C = np.linalg.solve(M, rhs)
