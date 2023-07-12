class ClassA:
    def __init__(self, a=1.0):
        self.a = a

    def multiply(self, lst):
        return [x*self.a for x in lst]
    
class ClassB:
    def __init__(self, b=2.0):
        self.b = b

    def process(self, lst):
        ca = ClassA()
        z = ca.multiply(lst)
        return [x+self.b for x in z]