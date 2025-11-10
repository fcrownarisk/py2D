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
    Formulas:
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
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

class GeometryVisualizer:
    """Visualize geometric shapes and coordinate systems"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def plot_triangle_square_pentagon(self):
        """Plot triangle, square, and pentagon"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Triangle
        triangle = Triangle(2)
        tri_points = triangle.points
        tri_x = [p.x for p in tri_points] + [tri_points[0].x]
        tri_y = [p.y for p in tri_points] + [tri_points[0].y]
        ax1.plot(tri_x, tri_y, 'r-', linewidth=2, label='Triangle')
        ax1.fill(tri_x, tri_y, 'red', alpha=0.3)
        ax1.set_title(f'Triangle (Area: {triangle.area():.2f})')
        ax1.grid(True)
        ax1.set_aspect('equal')
        
        # Square
        square = Square(2)
        sq_points = square.points
        sq_x = [p.x for p in sq_points] + [sq_points[0].x]
        sq_y = [p.y for p in sq_points] + [sq_points[0].y]
        ax2.plot(sq_x, sq_y, 'g-', linewidth=2, label='Square')
        ax2.fill(sq_x, sq_y, 'green', alpha=0.3)
        ax2.set_title(f'Square (Area: {square.area():.2f})')
        ax2.grid(True)
        ax2.set_aspect('equal')
        
        # Pentagon
        pentagon = Pentagon(2)
        pent_points = pentagon.points
        pent_x = [p.x for p in pent_points] + [pent_points[0].x]
        pent_y = [p.y for p in pent_points] + [pent_points[0].y]
        ax3.plot(pent_x, pent_y, 'b-', linewidth=2, label='Pentagon')
        ax3.fill(pent_x, pent_y, 'blue', alpha=0.3)
        ax3.set_title(f'Pentagon (Area: {pentagon.area():.2f})')
        ax3.grid(True)
        ax3.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def plot_coordinate_systems(self):
        """Plot 3D coordinate systems"""
        fig = plt.figure(figsize=(12, 5))
        
        # Rectangular to Spherical conversion demonstration
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Create some points in rectangular coordinates
        points_rect = [
            RectangularCoordinate(1, 0, 0),
            RectangularCoordinate(0, 1, 0),
            RectangularCoordinate(0, 0, 1),
            RectangularCoordinate(1, 1, 1)
        ]
        
        colors = ['red', 'green', 'blue', 'purple']
        for i, point in enumerate(points_rect):
            ax1.scatter(point.x, point.y, point.z, color=colors[i], s=100, label=f'Point {i+1}')
            
            # Convert to spherical and show relationship
            sphere_point = point.to_sphere()
            print(f"Point {i+1}: Rect({point.x}, {point.y}, {point.z}) -> "
                  f"Sphere(r={sphere_point.r:.2f}, θ={sphere_point.theta:.2f}, φ={sphere_point.phi:.2f})")
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Rectangular Coordinates')
        ax1.legend()
        
        # ARS Coordinate System
        ax2 = fig.add_subplot(122, projection='3d')
        
        ars = ARSCoordinateSystem()
        relationship_point = ars.find_relationship_point()
        ars.set_coordinates(*relationship_point)
        
        ax2.scatter(ars.rectangular.x, ars.rectangular.y, ars.rectangular.z, 
                   color='orange', s=200, label='X^5 = Y^4 = Z^3')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('ARS Coordinate Relationship Point')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """Main demonstration of the geometric and coordinate systems"""
    print("🔷 GEOMETRIC SHAPES AND COORDINATE SYSTEMS")
    print("=" * 60)
    
    # Create geometric shapes
    print("1. Creating Geometric Shapes:")
    print("-" * 30)
    
    triangle = Triangle(3)
    square = Square(4)
    pentagon = Pentagon(5)
    
    print(f"Triangle (edge={triangle.edge}):")
    print(f"  Area: {triangle.area():.4f}")
    print(f"  Perimeter: {triangle.perimeter():.4f}")
    
    print(f"\nSquare (edge={square.edge}):")
    print(f"  Area: {square.area():.4f}")
    print(f"  Perimeter: {square.perimeter():.4f}")
    
    print(f"\nPentagon (edge={pentagon.edge}):")
    print(f"  Area: {pentagon.area():.4f}")
    print(f"  Perimeter: {pentagon.perimeter():.4f}")
    
    # Check 345 relationship
    print(f"\n2. 345 Relationship Check:")
    print("-" * 30)
    is_345 = pythagorean_345(triangle, square, pentagon)
    print(f"Triangle² + Square² == Pentagon²: {is_345}")
    
    if is_345:
        print("  ✅ The 345 relationship holds!")
    else:
        print("  ❌ The 345 relationship does not hold with these edge lengths")
    
    # Coordinate system demonstrations
    print(f"\n3. Coordinate System Conversions:")
    print("-" * 30)
    
    # Test rectangular to spherical conversion
    test_points = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 1)
    ]
    
    for x, y, z in test_points:
        rect = RectangularCoordinate(x, y, z)
        sphere = rect.to_sphere()
        print(f"Rect({x}, {y}, {z}) -> Sphere(r={sphere.r:.4f}, θ={sphere.theta:.4f}, φ={sphere.phi:.4f})")
        
        # Convert back
        rect_back = sphere.to_rectangular()
        print(f"  Back to Rect: ({rect_back.x:.4f}, {rect_back.y:.4f}, {rect_back.z:.4f})")
    
    # ARS Coordinate System
    print(f"\n4. ARS Coordinate System:")
    print("-" * 30)
    
    ars_system = ARSCoordinateSystem()
    relationship_point = ars_system.find_relationship_point()
    ars_system.set_coordinates(*relationship_point)
    
    print(f"Point where X^5 == Y^4 == Z^3:")
    print(f"  Rectangular: ({ars_system.rectangular.x:.4f}, {ars_system.rectangular.y:.4f}, {ars_system.rectangular.z:.4f})")
    print(f"  Spherical: r={ars_system.spherical.r:.4f}, θ={ars_system.spherical.theta:.4f}, φ={ars_system.spherical.phi:.4f}")
    print(f"  Relationship holds: {ars_system.check_relationship()}")
    
    # Visualizations
    print(f"\n5. Creating Visualizations...")
    print("-" * 30)
    
    visualizer = GeometryVisualizer()
    visualizer.plot_triangle_square_pentagon()
    visualizer.plot_coordinate_systems()
    
    print(f"\n" + "=" * 60)
    print("GEOMETRIC SYSTEM TRANSLATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()