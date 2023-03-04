import math

# Classes
class Block:
    """
    Stores information about a Minecraft block.
    """
    def __init__(self, x, y, z, name):
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + "|" + self.name + ")"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z\
            
    def position(self) -> "Vector":
        return Vector(self.x, self.y, self.z)
    

class Vector:
    """
    Stores a 3D vector.
    """
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"

    def __sub__(self, other) -> "Vector":
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def magnitude(self) -> float:
        return math.sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))
    
    def direction(self) -> "Vector":
        mag = self.magnitude()
        return Vector(self.x / mag, self.y / mag, self.z / mag)