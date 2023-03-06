from TwodIBM.simulationtools import forceFluid,generate_circle
import numpy as np


def right_test():
    pass
def up_test():
    pass

def inflated_ball():
    pass
def static_test():
    pass

def force_test():
    print("DONE")
    xs, dtheta = generate_circle(.1, 100)
    stuff=forceFluid(.01,.01,100,100,xs,dtheta,1,k=np.ones)
    print("DONE")
force_test()