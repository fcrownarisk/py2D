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

class ElectromagneticFieldVisualizer:
    """Visualize electromagnetic fields and Maxwell's equations"""
    
    def __init__(self):
        self.maxwell = MaxwellEquations()
    
    def plot_electric_field_point_charge(self):
        """Plot electric field from a point charge (Gauss's Law demonstration)"""
        fig = plt.figure(figsize=(12, 5))
        
        # Create grid
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        z = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        
        # Point charge at origin
        q = 1e-9  # 1 nC
        epsilon_0 = 8.854e-12
        
        # Electric field components (2D slice)
        R = np.sqrt(X**2 + Y**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            Ex = (q * X) / (4 * np.pi * epsilon_0 * R**3)
            Ey = (q * Y) / (4 * np.pi * epsilon_0 * R**3)
        
        # Replace infinities with NaN
        Ex[R == 0] = np.nan
        Ey[R == 0] = np.nan
        
        # Plot electric field
        ax1 = fig.add_subplot(121)
        ax1.streamplot(X, Y, Ex, Ey, color='red', linewidth=1, density=2)
        ax1.scatter(0, 0, color='red', s=100, label='Positive Charge')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Electric Field from Point Charge\n(Gauss\'s Law)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_aspect('equal')
        
        # Plot field magnitude
        ax2 = fig.add_subplot(122, projection='3d')
        E_magnitude = np.sqrt(Ex**2 + Ey**2)
        surf = ax2.plot_surface(X, Y, E_magnitude, cmap='hot', alpha=0.8)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('|E| (V/m)')
        ax2.set_title('Electric Field Magnitude')
        
        plt.tight_layout()
        plt.show()
    
    def plot_magnetic_field_wire(self):
        """Plot magnetic field around a current-carrying wire (Ampere's Law)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Current along z-axis
        I = 1.0  # 1 Amp
        mu_0 = 4 * np.pi * 1e-7
        
        # Create grid
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, y)
        
        # Magnetic field components (Biot-Savart Law)
        R = np.sqrt(X**2 + Y**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            Bx = (-mu_0 * I * Y) / (2 * np.pi * R**2)
            By = (mu_0 * I * X) / (2 * np.pi * R**2)
        
        Bx[R == 0] = np.nan
        By[R == 0] = np.nan
        
        # Plot magnetic field
        ax1.streamplot(X, Y, Bx, By, color='blue', linewidth=1, density=2)
        ax1.scatter(0, 0, color='blue', s=100, marker='s', label='Current ⊙')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Magnetic Field around Current Wire\n(Ampere\'s Law)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_aspect('equal')
        
        # Plot field lines as circles
        theta = np.linspace(0, 2*np.pi, 100)
        for r in [0.3, 0.6, 0.9]:
            ax2.plot(r*np.cos(theta), r*np.sin(theta), 'b-', alpha=0.7)
        
        ax2.quiver(0.3, 0, 0, 0.3, scale=10, color='red')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Circular Magnetic Field Lines\n(∇·B = 0)')
        ax2.grid(True)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_wave_equation(self):
        """Show that Maxwell's equations lead to wave equation"""
        print("ELECTROMAGNETIC WAVE EQUATION DERIVATION")
        print("=" * 50)
        
        # Start with Maxwell's equations in vacuum (ρ=0, J=0)
        print("In vacuum (no charges, no currents):")
        print("∇·E = 0")
        print("∇·B = 0") 
        print("∇×E = -∂B/∂t")
        print("∇×B = μ₀ε₀∂E/∂t")
        print()
        
        # Take curl of Faraday's law
        print("Take curl of Faraday's law:")
        print("∇×(∇×E) = ∇×(-∂B/∂t)")
        print("∇(∇·E) - ∇²E = -∂/∂t (∇×B)")
        print("0 - ∇²E = -∂/∂t (μ₀ε₀∂E/∂t)")
        print("∇²E = μ₀ε₀ ∂²E/∂t²")
        print()
        
        # Wave equation!
        c = 1 / np.sqrt(MaxwellEquations.EPSILON_0 * MaxwellEquations.MU_0)
        print(f"WAVE EQUATION: ∇²E = (1/c²) ∂²E/∂t²")
        print(f"where c = 1/√(μ₀ε₀) = {c:.2e} m/s (speed of light!)")
        
        # Plot a simple electromagnetic wave
        self.plot_electromagnetic_wave()
    
    def plot_electromagnetic_wave(self):
        """Plot a simple electromagnetic wave"""
        fig = plt.figure(figsize=(15, 10))
        
        # Parameters
        wavelength = 1.0
        k = 2 * np.pi / wavelength  # wave number
        omega = MaxwellEquations.C * k  # angular frequency
        
        # Spatial grid
        x = np.linspace(0, 3 * wavelength, 100)
        z = np.linspace(0, 3 * wavelength, 100)
        X, Z = np.meshgrid(x, z)
        
        # Time instances
        times = [0, wavelength/(4*MaxwellEquations.C), wavelength/(2*MaxwellEquations.C)]
        titles = ['t = 0', 't = T/4', 't = T/2']
        
        for i, (t, title) in enumerate(zip(times, titles)):
            # Electric field (y-component)
            E_y = np.cos(k * X - omega * t)
            
            # Magnetic field (x and z components would be more complex in 2D)
            B_magnitude = E_y / MaxwellEquations.C
            
            ax = fig.add_subplot(2, 3, i+1)
            contour = ax.contourf(X, Z, E_y, levels=20, cmap='RdBu_r')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.set_title(f'Electric Field E_y\n{title}')
            plt.colorbar(contour, ax=ax)
            
            ax2 = fig.add_subplot(2, 3, i+4)
            contour2 = ax2.contourf(X, Z, B_magnitude, levels=20, cmap='RdBu_r')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Z (m)')
            ax2.set_title(f'Magnetic Field |B|\n{title}')
            plt.colorbar(contour2, ax=ax2)
        
        plt.tight_layout()
        plt.show()

class MaxwellEquationSolver:
    """Numerical solvers for Maxwell's equations in various scenarios"""
    
    def __init__(self):
        self.maxwell = MaxwellEquations()
    
    def solve_electrostatic(self, charge_distribution: Callable, bounds: Tuple, grid_size: int = 50):
        """
        Solve Poisson's equation for electrostatics: ∇²φ = -ρ/ε₀
        """
        print("SOLVING ELECTROSTATIC PROBLEM")
        print("Poisson's Equation: ∇²φ = -ρ/ε₀")
        print("Electric Field: E = -∇φ")
        print("=" * 50)
        
        # Create grid
        x = np.linspace(bounds[0], bounds[1], grid_size)
        y = np.linspace(bounds[2], bounds[3], grid_size)
        z = np.linspace(bounds[4], bounds[5], grid_size)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate charge density
        rho = charge_distribution(X, Y, Z)
        
        # Simple demonstration - for actual Poisson solver would need more sophisticated methods
        print(f"Charge distribution calculated on {grid_size}³ grid")
        print(f"Total charge: {np.sum(rho) * (x[1]-x[0])**3:.2e} C")
        
        return X, Y, Z, rho
    
    def point_charge_distribution(self, X, Y, Z):
        """Point charge at origin"""
        r = np.sqrt(X**2 + Y**2 + Z**2)
        rho = np.zeros_like(X)
        # Place charge near origin (avoid singularity)
        center_idx = X.shape[0] // 2
        rho[center_idx, center_idx, center_idx] = 1e-9  # 1 nC
        return rho

def demonstrate_maxwell_equations():
    """Comprehensive demonstration of all Maxwell's equations"""
    
    maxwell = MaxwellEquations()
    visualizer = ElectromagneticFieldVisualizer()
    solver = MaxwellEquationSolver()
    
    print("MAXWELL'S EQUATIONS COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    
    # Display all equations symbolically
    print("\n1. GAUSS'S LAW FOR ELECTRICITY:")
    print(maxwell.gauss_law_electricity('symbolic'))
    print(maxwell.gauss_law_electricity('physical'))
    
    print("\n2. GAUSS'S LAW FOR MAGNETISM:")
    print(maxwell.gauss_law_magnetism('symbolic'))
    print(maxwell.gauss_law_magnetism('physical'))
    
    print("\n3. FARADAY'S LAW OF INDUCTION:")
    print(maxwell.faradays_law('symbolic'))
    print(maxwell.faradays_law('physical'))
    
    print("\n4. AMPERE-MAXWELL LAW:")
    print(maxwell.amperes_law_with_maxwell('symbolic'))
    print(maxwell.amperes_law_with_maxwell('physical'))
    
    # Visualizations
    print("\n" + "="*60)
    print("VISUALIZATIONS")
    print("="*60)
    
    visualizer.plot_electric_field_point_charge()
    visualizer.plot_magnetic_field_wire()
    visualizer.demonstrate_wave_equation()
    
    # Numerical examples
    print("\n" + "="*60)
    print("NUMERICAL EXAMPLES")
    print("="*60)
    
    bounds = (-1, 1, -1, 1, -1, 1)
    X, Y, Z, rho = solver.solve_electrostatic(solver.point_charge_distribution, bounds, 20)
    
    # Display key physical insights
    print("\n" + "="*60)
    print("KEY PHYSICAL INSIGHTS")
    print("="*60)
    
    insights = """
    KEY INSIGHTS FROM MAXWELL'S EQUATIONS:
    
    1. SYMMETRY: 
       - Changing E produces B (Faraday)
       - Changing B produces E (Ampere-Maxwell)
    
    2. CONSERVATION LAWS:
       - Charge conservation: ∂ρ/∂t + ∇·J = 0
       - Energy conservation: Poynting theorem
    
    3. WAVE NATURE:
       - Self-sustaining electromagnetic waves
       - Prediction of light as EM wave
    
    4. RELATIVISTIC NATURE:
       - Lorentz invariant
       - Foundation of special relativity
    
    5. QUANTUM CONNECTION:
       - Basis for quantum electrodynamics (QED)
       - Photon as quantum of EM field
    """
    print(insights)

if __name__ == "__main__":
    demonstrate_maxwell_equations()