import math
import numpy as np
from typing import Tuple, List, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CoordinateSystem:
    """Base coordinate system class"""
    
    def __init__(self):
        pass

@dataclass
class Rect:
    """Rectangular coordinate point"""
    x: float
    y: float
    
    def __post_init__(self):
        self.coord_type = "Rectangular"

@dataclass
class Line:
    """Line coordinate point"""
    x: float
    y: float
    
    def __post_init__(self):
        self.coord_type = "Line"

@dataclass
class Dot:
    """Dot coordinate point with angle"""
    angle: float
    x: float
    y: float
    
    def __post_init__(self):
        self.coord_type = "Dot"

class Triangle:
    """Triangle geometric shape implementation"""
    
    def __init__(self, edge: float = 1.0):
        self.edge = edge
        self.points = self.calculate_points()
    
    def calculate_points(self) -> List[Rect]:
        """Calculate triangle vertices"""
        edge = self.edge
        dot1 = Rect(edge, edge)
        dot2 = Rect(-math.sqrt(3)/2 * edge, 0)
        dot3 = Rect(0, -math.sqrt(3)/2 * edge)
        return [dot1, dot2, dot3]
    
    def area(self) -> float:
        """Calculate triangle area"""
        return (math.sqrt(3) / 4) * self.edge ** 2
    
    def perimeter(self) -> float:
        """Calculate triangle perimeter"""
        return 3 * self.edge

class Square:
    """Square geometric shape implementation"""
    
    def __init__(self, edge: float = 1.0):
        self.edge = edge
        self.points = self.calculate_points()
    
    def calculate_points(self) -> List[Line]:
        """Calculate square vertices"""
        edge = self.edge
        dot4 = Line(edge, edge)
        dot5 = Line(-edge, edge)
        dot6 = Line(edge, -edge)
        dot7 = Line(-edge, -edge)
        return [dot4, dot5, dot6, dot7]
    
    def area(self) -> float:
        """Calculate square area"""
        return self.edge ** 2
    
    def perimeter(self) -> float:
        """Calculate square perimeter"""
        return 4 * self.edge

class Pentagon:
    """Pentagon geometric shape implementation"""
    
    def __init__(self, edge: float = 1.0):
        self.edge = edge
        self.points = self.calculate_points()
    
    def calculate_points(self) -> List[Dot]:
        """Calculate pentagon vertices using angles"""
        points = []
        for i in range(5):
            angle_deg = 72 * i  # 72 degrees per vertex (360/5)
            angle_rad = math.radians(angle_deg)
            x = self.edge * math.cos(angle_rad)
            y = self.edge * math.sin(angle_rad)
            points.append(Dot(angle_deg, x, y))
        return points
    
    def area(self) -> float:
        """Calculate pentagon area"""
        return (1/4) * math.sqrt(5 * (5 + 2 * math.sqrt(5))) * self.edge ** 2
    
    def perimeter(self) -> float:
        """Calculate pentagon perimeter"""
        return 5 * self.edge

def pythagorean_345(triangle: Triangle, square: Square, pentagon: Pentagon) -> bool:
    """
    Check the 345 relationship: Triangle^2 + Square^2 == Pentagon^2
    Using areas for the mathematical relationship
    """
    tri_area = triangle.area()
    sq_area = square.area()
    pent_area = pentagon.area()
    
    result = math.isclose(tri_area**2 + sq_area**2, pent_area**2, rel_tol=1e-9)
    return result

class AffineCoordinate:
    """Affine coordinate system implementation"""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.coord_type = "Affine"
    
    def __str__(self):
        return f"Affine({self.x}, {self.y}, {self.z})"

class RectangularCoordinate:
    """Rectangular (Cartesian) coordinate system implementation"""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.coord_type = "Rectangular"
    
    def to_sphere(self) -> 'SphereCoordinate':
        """Convert rectangular coordinates to spherical coordinates"""
        return rectangular_to_sphere(self.x, self.y, self.z)
    
    def __str__(self):
        return f"Rectangular({self.x}, {self.y}, {self.z})"

class SphereCoordinate:
    """Spherical coordinate system implementation"""
    
    def __init__(self, r: float = 0.0, theta: float = 0.0, phi: float = 0.0):
        self.r = float(r)
        self.theta = float(theta)  # azimuthal angle
        self.phi = float(phi)      # polar angle
        self.coord_type = "Spherical"
    
    def to_rectangular(self) -> RectangularCoordinate:
        """Convert spherical coordinates to rectangular coordinates"""
        return sphere_to_rectangular(self.r, self.theta, self.phi)
    
    def __str__(self):
        return f"Spherical(r={self.r}, θ={self.theta}, φ={self.phi})"

def rectangular_to_sphere(x: float, y: float, z: float) -> SphereCoordinate:
    """
    Convert rectangular coordinates to spherical coordinates
    Formulas:
    r = sqrt(x² + y² + z²)
    theta = atan2(y, x) [azimuthal angle]
    phi = atan2(sqrt(x² + y²), z) [polar angle]
    """
    r = math.sqrt(x*x + y*y + z*z)
    theta = math.atan2(y, x)
    phi = math.atan2(math.sqrt(x*x + y*y), z)
    
    return SphereCoordinate(r, theta, phi)

def sphere_to_rectangular(r: float, theta: float, phi: float) -> RectangularCoordinate:
    """
    Convert spherical coordinates to rectangular coordinates
    """
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    
    return RectangularCoordinate(x, y, z)

class ARSCoordinateSystem:
    """
    ARS Coordinate System: Affine, Rectangular, Spherical
    Implements the coordinate relationship: X^5 == Y^4 == Z^3
    """
    
    def __init__(self):
        self.affine = AffineCoordinate()
        self.rectangular = RectangularCoordinate()
        self.spherical = SphereCoordinate()
    
    def set_coordinates(self, x: float, y: float, z: float):
        """Set all coordinate systems from rectangular coordinates"""
        self.rectangular = RectangularCoordinate(x, y, z)
        self.spherical = self.rectangular.to_sphere()
        self.affine = AffineCoordinate(x, y, z)
    
    def check_relationship(self) -> bool:
        """
        Check the coordinate relationship: X^5 == Y^4 == Z^3
        Using the spherical coordinates for this relationship
        """
        x_pow = self.spherical.r ** 5
        y_pow = self.spherical.theta ** 4
        z_pow = self.spherical.phi ** 3
        
        # Check if all three are approximately equal
        return (math.isclose(x_pow, y_pow, rel_tol=1e-9) and 
                math.isclose(y_pow, z_pow, rel_tol=1e-9))
    
    def find_relationship_point(self) -> Tuple[float, float, float]:
        """
        Find a point where X^5 == Y^4 == Z^3
        This is a mathematical relationship we need to solve
        """
        # For simplicity, let's find when r^5 = theta^4 = phi^3 = k
        # We can choose k and work backwards
        k = 1.0  # Choose k=1 for simplicity
        r = k ** (1/5)
        theta = k ** (1/4)
        phi = k ** (1/3)
        
        # Convert back to rectangular coordinates
        rect_coord = sphere_to_rectangular(r, theta, phi)
        return rect_coord.x, rect_coord.y, rect_coord.z

    
    def __str__(self):
        
        return (f"ARS Coordinate System:\n"
                f"  Affine: {self.affine}\n"
                f"  Rectangular: {self.rectangular}\n"
                f"  Spherical: {self.spherical}\n"
                f"  Relationship X^5 == Y^4 == Z^3: {self.check_relationship()}")
