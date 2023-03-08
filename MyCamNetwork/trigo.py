import math

class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        return Vector(self.x + v.x, self.y + v.y)    

    def __mul__(self, v):
        return Vector(self.x * v.x, self.y * v.y)  
    
    def __sub__(self, v):
        return Vector(self.x - v.x, self.y - v.y)    

    def __truediv__(self, v):
        return Vector(self.x / v.x, self.y / v.y)    

    
def dot(a, b):
    return a.x*b.x+ a.y*b.y

def reflect(I, N):
    d = dot(N,I)
    return Vector(I.x - 2.0 * d * N.x, I.y - 2.0 * d * N.y)

def length(v):
    return math.sqrt(v.x**2 + v.y**2)

def distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def normalize(v):
    l = length(v)
    if l == 0.0:
        return Vector(0.0, 0.0)
    return Vector(v.x / l, v.y / l)