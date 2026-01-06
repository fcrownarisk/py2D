import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import sympy as sp
from sympy.vector import CoordSys3D, Del, curl, divergence
from typing import Tuple, Callable, Dict
import matplotlib.animation as animation

class MaxwellEquations:
    """
    Comprehensive implementation of Maxwell's 4 Equations
    """
    
    # Physical constants
    EPSILON_0 = 8.854187817e-12  # Vacuum permittivity [F/m]
    MU_0 = 4 * np.pi * 1e-7      # Vacuum permeability [H/m]
    C = 1 / np.sqrt(EPSILON_0 * MU_0)  # Speed of light [m/s]
    
    def __init__(self):
        self.setup_symbolic_system()
        
    def setup_symbolic_system(self):
        """Setup symbolic coordinate system and variables"""
        # Symbolic coordinate system
        self.R = CoordSys3D('R')
        self.x, self.y, self.z = self.R.x, self.R.y, self.R.z
        self.t = sp.Symbol('t', real=True)
        
        # Del operator
        self.del_op = Del()
        
        # Symbolic field components
        self.Ex = sp.Function('E_x')(self.x, self.y, self.z, self.t)
        self.Ey = sp.Function('E_y')(self.x, self.y, self.z, self.t)
        self.Ez = sp.Function('E_z')(self.x, self.y, self.z, self.t)
        
        self.Bx = sp.Function('B_x')(self.x, self.y, self.z, self.t)
        self.By = sp.Function('B_y')(self.x, self.y, self.z, self.t)
        self.Bz = sp.Function('B_z')(self.x, self.y, self.z, self.t)
        
        # Symbolic vector fields
        self.E_field = self.Ex * self.R.i + self.Ey * self.R.j + self.Ez * self.R.k
        self.B_field = self.Bx * self.R.i + self.By * self.R.j + self.Bz * self.R.k
        
        # Charge and current densities
        self.rho = sp.Function('rho')(self.x, self.y, self.z, self.t)  # Charge density
        self.Jx = sp.Function('J_x')(self.x, self.y, self.z, self.t)   # Current density components
        self.Jy = sp.Function('J_y')(self.x, self.y, self.z, self.t)
        self.Jz = sp.Function('J_z')(self.x, self.y, self.z, self.t)
        self.J_field = self.Jx * self.R.i + self.Jy * self.R.j + self.Jz * self.R.k

    def gauss_law_electricity(self, method: str = 'symbolic') -> str:
        """
        Gauss's Law for Electricity: ∇·E = ρ/ε₀
        
        The electric flux through any closed surface is proportional 
        to the enclosed electric charge.
        """
        if method == 'symbolic':
            lhs = divergence(self.E_field).doit()
            rhs = self.rho / self.EPSILON_0
            return f"∇·E = {lhs}\nρ/ε₀ = {rhs}"
        
        elif method == 'physical':
            explanation = """
            GAUSS'S LAW FOR ELECTRICITY:
            ∇·E = ρ/ε₀
            
            PHYSICAL MEANING:
            - Electric field lines begin on positive charges and end on negative charges
            - The divergence of E at a point equals the charge density at that point divided by ε₀
            - For a closed surface, the total electric flux equals Q_enc/ε₀
            
            MATHEMATICAL FORM:
            ∂E_x/∂x + ∂E_y/∂y + ∂E_z/∂z = ρ(x,y,z,t) / ε₀
            
            APPLICATIONS:
            - Calculating E fields from symmetric charge distributions
            - Understanding electric field behavior in materials
            - Fundamental to electrostatics
            """
            return explanation
    
    def gauss_law_magnetism(self, method: str = 'symbolic') -> str:
        """
        Gauss's Law for Magnetism: ∇·B = 0
        
        There are no magnetic monopoles - magnetic field lines always form closed loops.
        """
        if method == 'symbolic':
            lhs = divergence(self.B_field).doit()
            return f"∇·B = {lhs} = 0"
        
        elif method == 'physical':
            explanation = """
            GAUSS'S LAW FOR MAGNETISM:
            ∇·B = 0
            
            PHYSICAL MEANING:
            - There are no magnetic monopoles (no isolated North or South poles)
            - Magnetic field lines always form closed loops
            - The net magnetic flux through any closed surface is zero
            
            MATHEMATICAL FORM:
            ∂B_x/∂x + ∂B_y/∂y + ∂B_z/∂z = 0
            
            CONSEQUENCES:
            - Magnetic poles always come in North-South pairs
            - Magnetic field is solenoidal (divergence-free)
            - Fundamental property of the magnetic field
            """
            return explanation
    
    def faradays_law(self, method: str = 'symbolic') -> str:
        """
        Faraday's Law of Induction: ∇×E = -∂B/∂t
        
        A changing magnetic field produces an electric field.
        """
        if method == 'symbolic':
            lhs = curl(self.E_field).doit()
            rhs = -sp.Derivative(self.B_field, self.t)
            return f"∇×E = {lhs}\n-∂B/∂t = {rhs}"
        
        elif method == 'physical':
            explanation = """
            FARADAY'S LAW OF INDUCTION:
            ∇×E = -∂B/∂t
            
            PHYSICAL MEANING:
            - A changing magnetic field induces an electromotive force (EMF)
            - The induced EMF creates an electric field that circulates around changing B
            - Basis for generators, transformers, and induction
            
            MATHEMATICAL FORM (Component-wise):
            (∂E_z/∂y - ∂E_y/∂z) = -∂B_x/∂t
            (∂E_x/∂z - ∂E_z/∂x) = -∂B_y/∂t  
            (∂E_y/∂x - ∂E_x/∂y) = -∂B_z/∂t
            
            INTEGRAL FORM:
            ∮ E·dl = -d/dt ∫ B·dA  (EMF = -rate of change of magnetic flux)
            """
            return explanation
    
    def amperes_law_with_maxwell(self, method: str = 'symbolic') -> str:
        """
        Ampere-Maxwell Law: ∇×B = μ₀J + μ₀ε₀∂E/∂t
        
        Electric currents and changing electric fields produce magnetic fields.
        """
        if method == 'symbolic':
            lhs = curl(self.B_field).doit()
            rhs = self.MU_0 * self.J_field + self.MU_0 * self.EPSILON_0 * sp.Derivative(self.E_field, self.t)
            return f"∇×B = {lhs}\nμ₀J + μ₀ε₀∂E/∂t = {rhs}"
        elif method == 'physical':
            explanation = """
            AMPERE-MAXWELL LAW:
            ∇×B = μ₀J + μ₀ε₀∂E/∂t
            
            PHYSICAL MEANING:
            - Electric currents (J) produce circulating magnetic fields
            - Changing electric fields (displacement current) also produce magnetic fields
            - Completes the symmetry with Faraday's law
            
            MATHEMATICAL FORM (Component-wise):
            (∂B_z/∂y - ∂B_y/∂z) = μ₀J_x + μ₀ε₀∂E_x/∂t
            (∂B_x/∂z - ∂B_z/∂x) = μ₀J_y + μ₀ε₀∂E_y/∂t
            (∂B_y/∂x - ∂B_x/∂y) = μ₀J_z + μ₀ε₀∂E_z/∂t
            
            DISPLACEMENT CURRENT:
            The term μ₀ε₀∂E/∂t is Maxwell's crucial addition - the displacement current
            """
            return explanation
if __name__ == "__main__":
    demonstrate_maxwell_equations()
