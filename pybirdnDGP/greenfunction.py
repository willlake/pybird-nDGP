from scipy.integrate import solve_ivp
from scipy.integrate import quad, quad_vec
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint
from scipy.special import erf
from numpy import exp,log,linspace,array,sqrt,pi
import numpy as np

class GreenFunction(object):
   
    def __init__(self, Omega0_m, Omega0_rc=None, w0=-1.0,wa=0.0, H0=70, x0=-7.,x1=0.,nxgrid=500,vectorize=False,need_growth=True):
        self.vectorize = vectorize
        self.Omega0_m = Omega0_m
        self.Omega0_rc = Omega0_rc
        self.OmegaL_by_Omega_m = (1.-self.Omega0_m)/self.Omega0_m
        self.H0 = H0

        if Omega0_rc is not None:
            self.Omega0_rc = Omega0_rc
            self.nDGP = True
        else:
            self.nDGP = False
            
        self.x1  = x1 # the final time used to set initial condition for decay mode
        self.x0  = x0 # the initial time for ODE
        self.xlo = x0  # the lowest integration bound of scale factor x, x=lna
        # precsion for odeint
        self.rtol=1e-8 
        self.atol=1e-10
        self.ngrid = nxgrid
        self.w0 = w0
        self.wa = wa
        self.expansion_model = "w0wa"

        self.epsrel = 1e-6
        if need_growth:
            self.get_DDD_plusminus()

    def beta(self, x):
        r"""
        beta(a) = 1 + (H(a)/H0) * 1/sqrt(Omega0_rc) * ( 1 + aH'(a) / (3 H(a)) )
    
        Here x = ln a, and H'(a) means dH/d a, so:
            a H'(a) = dH/dlna = self.dHdx(x).
        """
        H = self.H(x)
        Hp = self.dHdx(x)  # dH/d ln a
        return 1.0 + (H / self.H0) * (1.0 / np.sqrt(self.Omega0_rc)) * (1.0 + Hp / (3.0 * H))
        

    def mu_Phi(self, x):
        r"""
        nu(a) = 1 + 1 / (3 beta(a))
        """
        if self.nDGP:
            b = self.beta(x)
            return 1.0 + 1.0 / (3.0 * b)
        else:
            return 1
    
    
    def mu2(self, x):
        r"""
        nu2(a) = -1/2 * (H/H0)^2 * 1/Omega0_rc * (1/(3 beta))^3
        """
        if self.nDGP:
            b = self.beta(x)
            H_over_H0 = self.H(x) / self.H0
            inv3b = 1.0 / (3.0 * b)
            return -0.5 * (H_over_H0**2) * (1.0 / self.Omega0_rc) * (inv3b**3)
        else:
            return 0
    
    
    def mu22(self, x):
        r"""
        nu22(a) = 2 * (H/H0)^4 * 1/Omega0_rc^2 * (1/(3 beta))^5
        """
        if self.nDGP:
            b = self.beta(x)
            H_over_H0 = self.H(x) / self.H0
            inv3b = 1.0 / (3.0 * b)
            return 2.0 * (H_over_H0**4) * (1.0 / (self.Omega0_rc**2)) * (inv3b**5)
        else:
            return 0

    def a_of_x(self, x):
        return np.exp(x)
    
    def w(self, x):
        if self.expansion_model != "w0wa":
            raise NotImplementedError
        a = self.a_of_x(x)
        return self.w0 + self.wa * (1.0 - a)
    
    def dwdx(self, x):
        if self.expansion_model != "w0wa":
            raise NotImplementedError
        a = self.a_of_x(x)
        return -self.wa * a
    
    def f_de(self, x):
        """rho_DE(a)/rho_DE,0 for CPL (exact)."""
        if self.expansion_model != "w0wa":
            raise NotImplementedError
        a = self.a_of_x(x)
        w0, wa = self.w0, self.wa
        return a**(-3.0*(1.0 + w0 + wa)) * np.exp(3.0*wa*(a - 1.0))
    
    def E2(self, x):
        """E^2 = (H/H0)^2 for flat matter + DE only."""
        a = self.a_of_x(x)
        Om0 = self.Omega0_m
        return Om0 * a**(-3.0) + (1.0 - Om0) * self.f_de(x)
    
    def H(self, x):
        return self.H0 * np.sqrt(self.E2(x))
    
    def Om(self, x):
        a = self.a_of_x(x)
        return self.Omega0_m * a**(-3.0) / self.E2(x)
    
    def Ode(self, x):
        # flat: Ode = 1 - Om also works (numerically stable and cheaper)
        return 1.0 - self.Om(x)
    
    def dOmdx(self, x):
        w = self.w(x)
        Om = self.Om(x)
        Ode = 1.0 - Om
        return 3.0 * w * Om * Ode
    
    def dOdedx(self, x):
        return -self.dOmdx(x)
    
    def dHdx(self, x):
        H = self.H(x)
        Ode = self.Ode(x)
        w = self.w(x)
        # d ln H / dx = -3/2 (1 + w*Ode)
        return -1.5 * H * (1.0 + w * Ode)
    
    

    def linear_growth_equation(self, y, x):
        """
        Linear growth ODE as a first-order system in x = ln a.
    
        y = (dD/dx, D)
        returns (d^2D/dx^2, dD/dx)
        """
        dD_dx, D = y
    
        H = self.H(x)
        dlnH_dx = self.dHdx(x) / H          # d ln H / dx
        Fx = dlnH_dx + 2.0              # Friction term:(2 + d ln H/dx)
    
        mu = self.mu_Phi(x) if self.nDGP else 1.0
        Om = self.Om(x)
        Sx = 1.5 * Om * mu     # Source term:(3/2) Omega_m * mu
    
        d2D_dx2 = -Fx * dD_dx + Sx * D
        return (d2D_dx2, dD_dx)

    def get_DDD_plusminus(self):
        """
        Solve growing mode D_+ with IC D=a at x0, x=lna, then construct decaying mode D_- from D_+.
        """
    
        # --- grid where we'll evaluate everything (useful for stable quadratures) ---
        x0, x1 = self.x0, self.x1
        x_grid = np.linspace(self.x0, self.x1, self.ngrid)
        a0 = np.exp(self.x0)
    
        # --- initial conditions for growing mode: D=a, dD/dx = a (since D=a => dD/dx = d(a)/dx = a) ---
        y0 = (a0, a0)  # (dD/dx, D)
    
        sol = solve_ivp(
            fun=lambda x, y: self.linear_growth_equation(y, x),
            t_span=(x0, x1),
            y0=y0,
            t_eval=x_grid,
            rtol=self.rtol,
            atol=self.atol,
            method="RK45",
        )
    
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
    
        # from solve_ivp in x
        D_plus     = sol.y[1]
        dDplus_dx  = sol.y[0]
        
        self.D  = CubicSpline(x_grid, D_plus)
        self.DD = CubicSpline(x_grid, dDplus_dx)     # NOW this is dD/dx (NOT dD/da)

        Dp2 = D_plus**2

        # integrating factor E(x) = exp(-∫ F dx)
        E = self.exp_minus_intF(x_grid)
        
        # g(x) = E / D_+^2
        g = E / Dp2
        
        # I(x) = ∫_{x0}^{x} g(u) du
        I = cumulative_trapezoid(g, x_grid, initial=0.0)
        
        # Remove the arbitrary constant to avoid D_+ contamination:
        # J(x) = I(x) - I(x1)  => D2(x1)=0
        J = I - I[-1]
        
        # Second independent solution
        D2 = D_plus * J
        
        # derivative wrt x = ln a
        dD2_dx = dDplus_dx * J + D_plus * g
        
        # store as "minus" slot (or make new attributes)
        self.Dminus  = CubicSpline(x_grid, D2)
        self.DDminus = CubicSpline(x_grid, dD2_dx)   # dD2/dx

    def exp_minus_intF(self, x):
        """
        Compute exp(-∫_{x0}^{x} F(s) ds) analytically for
          F(x) = 2 + d ln H / dx.
    
        Result:
          exp(-∫ F ds) = exp(-2 (x - x0)) * H(x0)/H(x)
        """
        x0  = self.x0
        Hx0 = self.H(x0)
        Hx  = self.H(x)
        return np.exp(-2.0 * (x - x0)) * (Hx0 / Hx)
        
    
    # ============================================================
    # Linear building blocks in x = ln a
    # ============================================================
    
    def fplus(self, x):
        # f = d ln D / d ln a = D'(x)/D(x)
        return self.DD(x) / self.D(x)
    
    def fminus(self, x):
        return self.DDminus(x) / self.Dminus(x)
    
    def W(self, x):
        # W_x = D * Dminus' - D' * Dminus
        return self.D(x) * self.DDminus(x) - self.DD(x) * self.Dminus(x)
    
    
    # ============================================================
    # Green's functions in (x, u) where u = ln ai
    # IMPORTANT: original a-space formulas have factors of ai (or ai^2).
    # Under ai = e^u, those become exp(u), exp(2u).
    # ============================================================
    
    def W(self, x):
        # W_x(x) = D_+(x) D_-'(x) - D_+'(x) D_-(x)
        return self.D(x) * self.DDminus(x) - self.DD(x) * self.Dminus(x)
    
    def G1d(self, x, u):
        pref = 1.0 / self.W(u)
        num  = self.DDminus(u) * self.D(x) - self.DD(u) * self.Dminus(x)
        return pref * num 
    
    def G2d(self, x, u):
        pref = self.fplus(u) / self.W(u)
        num  = self.D(u) * self.Dminus(x) - self.Dminus(u) * self.D(x)
        return pref * num 
    
    def G1t(self, x, u):
        pref = 1.0 / (self.fplus(x) * self.W(u))
        num  = self.DDminus(u) * self.DD(x) - self.DD(u) * self.DDminus(x)
        return pref * num 
    
    def G2t(self, x, u):
        pref = self.fplus(u) / (self.fplus(x) * self.W(u))
        num  = self.D(u) * self.DDminus(x) - self.Dminus(u) * self.DD(x)
        return pref * num 
    
    
    # ============================================================
    # Second-order integrands I1*, I2* in (u, x)
    # In a-space you had I*(ai,a) and then integrate dai.
    # In x-space: integrate du with weight exp(u) (since dai = exp(u) du).
    # We'll bake that weight into I* to keep quad() simple.
    # ============================================================
    
    def I1d(self, u, x):
        # corresponds to original I1d(ai, a)
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
    
        if self.nDGP:
            term = (self.G1d(x, u) * fp
                    + self.G2d(x, u) * self.mu2(u) * (1.5 * self.Om(u))**2 / fp)
        else:
            term = fp * self.G1d(x, u)
    
        # multiply by (D(u)^2 / D(x)^2) and Jacobian dai = exp(u) du
        return term * (Du*Du) / (Dx*Dx) 
    
    def I2d(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
    
        if self.nDGP:
            term = self.G2d(x, u) * (fp - self.mu2(u) * (1.5 * self.Om(u))**2 / fp)
        else:
            term = fp * self.G2d(x, u)
    
        return term * (Du*Du) / (Dx*Dx) 
    
    def I1t(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
    
        if self.nDGP:
            term = (self.G1t(x, u) * fp
                    + self.G2t(x, u) * self.mu2(u) * (1.5 * self.Om(u))**2 / fp)
        else:
            term = fp * self.G1t(x, u)
    
        return term * (Du*Du) / (Dx*Dx) 
    
    def I2t(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
    
        if self.nDGP:
            term = self.G2t(x, u) * (fp - self.mu2(u) * (1.5 * self.Om(u))**2 / fp)
        else:
            term = fp * self.G2t(x, u)
    
        return term * (Du*Du) / (Dx*Dx) 
    
    
    # ============================================================
    # Second-order time integrals (functions of x)
    # Original: ∫_{lo}^{a} I(ai,a) dai
    # Now:      ∫_{xlo}^{x} I(u,x) du   (with I already including exp(u))
    # ============================================================
    
    def mG1d(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.I1d(u, x), self.xlo, x,
                            epsrel=self.epsrel, epsabs=1.49e-08)[0]
        return quad(lambda u: self.I1d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mG2d(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.I2d(u, x), self.xlo, x,
                            epsrel=self.epsrel, epsabs=1.49e-08)[0]
        return quad(lambda u: self.I2d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mG1t(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.I1t(u, x), self.xlo, x,
                            epsrel=self.epsrel, epsabs=1.49e-08)[0]
        return quad(lambda u: self.I1t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mG2t(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.I2t(u, x), self.xlo, x,
                            epsrel=self.epsrel, epsabs=1.49e-08)[0]
        return quad(lambda u: self.I2t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def G(self, x):
        return self.mG1d(x) + self.mG2d(x)
    
    
    # ============================================================
    # Third-order integrands (same pattern): bake exp(u) Jacobian in
    # ============================================================
    
    def IU1d(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G1d(x, u) * fp * self.mG1d(u)
                    + self.G2d(x, u) * (1.5 * self.Om(u))**2
                      * (self.mu2(u) * self.mG1d(u) + 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp)
        else:
            term = fp * self.mG1d(u) * self.G1d(x, u)
    
        return term * ratio3 
    
    def IU2d(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G1d(x, u) * fp * self.mG2d(u)
                    + self.G2d(x, u) * (1.5 * self.Om(u))**2
                      * (self.mu2(u) * self.mG2d(u) - 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp)
        else:
            term = fp * self.mG2d(u) * self.G1d(x, u)
    
        return term * ratio3 
    
    def IU1t(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G1t(x, u) * fp * self.mG1d(u)
                    + self.G2t(x, u) * (1.5 * self.Om(u))**2
                      * (self.mu2(u) * self.mG1d(u) + 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp)
        else:
            term = fp * self.mG1d(u) * self.G1t(x, u)
    
        return term * ratio3 
    
    def IU2t(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G1t(x, u) * fp * self.mG2d(u)
                    + self.G2t(x, u) * (1.5 * self.Om(u))**2
                      * (self.mu2(u) * self.mG2d(u) - 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp)
        else:
            term = fp * self.mG2d(u) * self.G1t(x, u)
    
        return term * ratio3 
    
    
    def IV11d(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G1d(x, u) * fp * self.mG1t(u)
                    + self.G2d(x, u) * (1.5 * self.Om(u))**2
                      * (self.mu2(u) * self.mG1d(u) + 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp)
        else:
            term = fp * self.mG1t(u) * self.G1d(x, u)
    
        return term * ratio3 
    
    def IV12d(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G2d(x, u) * (fp * self.mG1t(u)
                    - (1.5 * self.Om(u))**2 * (self.mu2(u) * self.mG1d(u) + 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp))
        else:
            term = fp * self.mG1t(u) * self.G2d(x, u)
    
        return term * ratio3 
    
    def IV21d(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G1d(x, u) * fp * self.mG2t(u)
                    + self.G2d(x, u) * (1.5 * self.Om(u))**2
                      * (self.mu2(u) * self.mG2d(u) - 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp)
        else:
            term = fp * self.mG2t(u) * self.G1d(x, u)
    
        return term * ratio3 
    
    def IV22d(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G2d(x, u) * (fp * self.mG2t(u)
                    - (1.5 * self.Om(u))**2 * (self.mu2(u) * self.mG2d(u) - 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp))
        else:
            term = fp * self.mG2t(u) * self.G2d(x, u)
    
        return term * ratio3 
    
    
    def IV11t(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G1t(x, u) * fp * self.mG1t(u)
                    + self.G2t(x, u) * (1.5 * self.Om(u))**2
                      * (self.mu2(u) * self.mG1d(u) + 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp)
        else:
            term = fp * self.mG1t(u) * self.G1t(x, u)
    
        return term * ratio3 
    
    def IV12t(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G2t(x, u) * (fp * self.mG1t(u)
                    - (1.5 * self.Om(u))**2 * (self.mu2(u) * self.mG1d(u) + 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp))
        else:
            term = fp * self.mG1t(u) * self.G2t(x, u)
    
        return term * ratio3 
    
    def IV21t(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G1t(x, u) * fp * self.mG2t(u)
                    + self.G2t(x, u) * (1.5 * self.Om(u))**2
                      * (self.mu2(u) * self.mG2d(u) - 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp)
        else:
            term = fp * self.mG2t(u) * self.G1t(x, u)
    
        return term * ratio3 
    
    def IV22t(self, u, x):
        Du, Dx = self.D(u), self.D(x)
        fp = self.fplus(u)
        ratio3 = (Du / Dx)**3
    
        if self.nDGP:
            term = (self.G2t(x, u) * (fp * self.mG2t(u)
                    - (1.5 * self.Om(u))**2 * (self.mu2(u) * self.mG2d(u) - 0.5 * self.mu22(u) * 1.5 * self.Om(u)) / fp))
        else:
            term = fp * self.mG2t(u) * self.G2t(x, u)
    
        return term * ratio3 
    
    
    # ============================================================
    # Third-order time integrals: ∫_{xlo}^{x} du (integrand already has exp(u))
    # ============================================================
    
    def mU1d(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IU1d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IU1d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mU2d(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IU2d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IU2d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mU1t(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IU1t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IU1t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mU2t(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IU2t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IU2t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    
    def mV11d(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IV11d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IV11d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mV12d(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IV12d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IV12d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mV21d(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IV21d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IV21d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mV22d(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IV22d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IV22d(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    
    def mV11t(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IV11t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IV11t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mV12t(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IV12t(u, x), self.xlo, x, epsrel=self.epsrel, epsabs=1.49e-5)[0]
        return quad(lambda u: self.IV12t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mV21t(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IV21t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IV21t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    def mV22t(self, x):
        if self.vectorize:
            return quad_vec(lambda u: self.IV22t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
        return quad(lambda u: self.IV22t(u, x), self.xlo, x, epsrel=self.epsrel)[0]
    
    
    def Y(self, x):
        return -3/14. + self.mV11d(x) + self.mV12d(x)