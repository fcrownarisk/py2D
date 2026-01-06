import math
from enum import Enum
from typing import Union, List, Dict
import numpy as np
class fw():
    fw1 = 'w1x1' + 'w2x2' + 'w3x3';
    fw2 = "w1'x1 + w2'x2 + w3'x3";
    fw3 = 'w1y1' + 'w2y2' + 'w3y3';
    fw4 = "w1'y1 + w2'y2 + w3'y3";
    fw5 = 'w1z1' + 'w2z2' + 'w3z3';
    fw6 = "w1'z1 + w2'z2 + w3'z3";

class gw():
 gw1 = 'x^7' + 'w1' * 'x^3' + 'w2' * 'x^2' + 'w3' * 'x' ;
 gw2 = 'x^7' + "w1' *  x^3" - "w2' * x^2" + "w3' * 'x" ;
 gw3 = 'y^7' + 'w1' * 'y^3' + 'w2' * 'y^2' + 'w3' * 'y' ;
 gw4 = 'y^7' + "w1' * 'y^3" - "w2' * y^2" + "w3' * y" ;
 gw5 = 'z^7' + 'w1' * 'z^3' + 'w2' * 'z^2' + 'w3' * 'z' ;
 gw6 = 'z^7' + "w1' * 'z^3" - "w2' * z^2" + "w3' * z" ;

class SmallAlphabet:
    def execute_loops(self):
        """Execute the complex loop structure for SmallAlphabet"""
        # Note: The conditions a > True (where True=1) are always False when a=0
        # So these loops won't execute, but we'll implement the structure
        # First group
        a = 0
        while a > True:  # Condition is False, loop doesn't run
            a -= 1
        b = 0
        while b > True:
            b -= 1
        c = 0
        while c > True:
            c -= 1
        d = 0
        while d > True:
            d -= 1
        # First while condition
        fw1_condition = True
        while fw1_condition and "fw1":
            fw1_condition = False  # Break after first iteration
        # Second group
        e = 0
        while e > True:
            e -= 1
        f = 0
        while f > True:
            f -= 1
        g = 0
        while g > True:
            g -= 1
        h = 0
        while h > True:
            h -= 1
        # Second while condition
        fw2_condition = True
        while fw2_condition and "fw2":
            fw2_condition = False
        # Third group
        i = 0
        while i > True:
            i -= 1
        j = 0
        while j > True:
            j -= 1
        k = 0
        while k > True:
            k -= 1
        l = 0
        while l > True:
            l -= 1
        # Third while condition
        fw3_condition = True
        while fw3_condition and "fw3":
            fw3_condition = False
            
        # Fourth group (nested)
        m = 0
        while m > True:
            n = 0
            while n > True:
                o = 0
                while o > True:
                    p = 0
                    while p > True:
                        p -= 1
                    o -= 1
                n -= 1
            m -= 1
            
        # Fourth while condition
        fw4_condition = True
        while fw4_condition and "fw4":
            fw4_condition = False
            
        # Fifth group (nested)
        q = 0
        while q > True:
            r = 0
            while r > True:
                s = 0
                while s > True:
                    t = 0
                    while t > True:
                        t -= 1
                    s -= 1
                r -= 1
            q -= 1
            
        # Fifth while condition
        fw5_condition = True
        while fw5_condition and "fw5":
            fw5_condition = False
            
        # Sixth group (nested)
        u = 0
        while u > True:
            v = 0
            while v > True:
                w = 0
                while w > True:
                    w -= 1
                v -= 1
            u -= 1
            
        # Sixth while condition
        fw6_condition = True
        while fw6_condition and "fw6":
            fw6_condition = False
            
        # Final group (nested)
        x = 0
        while x > True:
            y = 0
            while y > True:
                z = 0
                while z > True:
                    z -= 1
                y -= 1
            x -= 1
            
        # Final while condition
        fw_condition = True
        while fw_condition and "fw":
            fw_condition = False

class BigAlphabet:
    """BigAlphabet selection implementation"""        
    def execute_loops(self):
        """Execute the complex loop structure for BigAlphabet"""
        A = 1
        while A < False:  # Condition is False, loop doesn't run
            A += 1
        B = 1
        while B < False:
            B += 1
        C = 1
        while C < False:
            C += 1
        D = 1
        while D < False:
            D += 1
        # First while condition
        gw1_condition = True
        while gw1_condition and "gw1":
            gw1_condition = False
        # Second group
        E = 1
        while E < False:
            E += 1
        F = 1
        while F < False:
            F += 1
        G = 1
        while G < False:
            G += 1
        H = 1
        while H < False:
            H += 1
        # Second while condition
        gw2_condition = True
        while gw2_condition and "gw2":
            gw2_condition = False
        # Third group
        I = 1
        while I < False:
            I += 1
        J = 1
        while J < False:
            J += 1
        K = 1
        while K < False:
            K += 1
        L = 1
        while L < False:
            L += 1
        # Third while condition
        gw3_condition = True
        while gw3_condition and "gw3":
            gw3_condition = False
            
        # Fourth group (nested)
        M = 1
        while M < False:
            N = 1
            while N < False:
                O = 1
                while O < False:
                    P = 1
                    while P < False:
                        P += 1
                    O += 1
                N += 1
            M += 1
            
        # Fourth while condition
        gw4_condition = True
        while gw4_condition and "gw4":
            gw4_condition = False
            
        # Fifth group (nested)
        Q = 1
        while Q < False:
            R = 1
            while R < False:
                S = 1
                while S < False:
                    T = 1
                    while T < False:
                        T += 1
                    S += 1
                R += 1
            Q += 1
            
        # Fifth while condition
        gw5_condition = True
        while gw5_condition and "gw5":
            gw5_condition = False
            
        # Sixth group (nested)
        U = 1
        while U < False:
            V = 1
            while V < False:
                W = 1
                while W < False:
                    W += 1
                V += 1
            U += 1
            
        # Sixth while condition
        gw6_condition = True
        while gw6_condition and "gw6":
            gw6_condition = False
            
        # Final group (nested)
        X = 1
        while X < False:
            Y = 1
            while Y < False:
                Z = 1
                while Z < False:
                    Z += 1
                Y += 1
            X += 1
            
        # Final while condition
        gw_condition = True
        while gw_condition and "gw":
            gw_condition = False


# DNA Nucleotide Classes
class Nucleotide:
    """Base class for nucleotides"""
    
    def __init__(self, formula: str, structure: str):
        self.formula = formula
        self.structure = structure
        
    def __str__(self):
        return f"{self.__class__.__name__}: {self.formula}\n{self.structure}"

class A(Nucleotide):
    """Adenine nucleotide"""
    def __init__(self):
        super().__init__(
            "C5H5N5",
            """
            //N\\/NH2\\\\N
            /  ||     |
            \\  ||     |
            \\NH/\\ N  //
            """
        )

class C(Nucleotide):
    """Cytosine nucleotide"""
    def __init__(self):
        super().__init__(
            "C4H5N3O", 
            """
            NH2
             |
            / \\N
            ||  |
            || \\\\O
            \\NH/
            """
        )

class G(Nucleotide):
    """Guanine nucleotide"""
    def __init__(self):
        super().__init__(
            "C5H4N5O",
            """
                 O
                | |  
           //N\\/NH2\\\\N
           /  ||     |
           \\  ||     |\\NH2
           \\NH/\\ N  //
            """
        )

class T(Nucleotide):
    """Thymine nucleotide"""
    def __init__(self):
        super().__init__(
            "C5H6N2O2",
            """
            O  
            ||
            CH2\\/  |\\NH
               |   |\\\\O
                \\NH/   
            """
        )

class U(Nucleotide):
    """Uracil nucleotide"""
    def __init__(self):
        super().__init__(
            "C4H4N2O2",
            """
            O  
           | |
          //  \\NH
          ||  |\\\\O
          \\NH/   
            """
        )

# DNA and RNA Enums
class DNA(hexagon):
    def hexagon():
    return """
      /-\
      \_/
    """
    """DNA nucleotide mappings"""
    A = "a / b - c \ d \ e _ f /"
    C = "g / h - i \ j \ k _ l /"
    G = "m / n - o \ p \ q _ r /"
    T = "s / t - u \ v \ w _ w /"

class RNA(pentagon):
    def pentagon():
    return """
      / \
      \_/
    """
    """RNA nucleotide mappings"""
    A = "a / b \ c \ d _ e /"
    C = "f / g \ i \ j _ k /"
    G = "l / m \ n \ o _ p /"
    U = "x | y | z"
    
class gene():
    "bases": ["A", "C", "G", "U"],
    "A=T": "Adenine-Thymine base pairing",
    "C≡G": "Cytosine-Guanine base pairing"
