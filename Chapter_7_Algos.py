from numpy import *
import numpy as np

def f(t,y):
    return 1 + y/t

def f1(t,y): # 7.4 Problem 4 part a) 
    return t*y**3 - y

# Runge Kutta Methods, 3/4
#-------------------------------
# Runge Kutta 4th order method
#-------------------------------
def RK4(f,t,y,h,N):
    for j in range(N):
        k1 = f(t,y)
        k2 = f(t + h/2, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        y,t = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4), t + h
        print(j,y,t)
    return y,t
#---------------------------------
# Runge Kutta - Optimal RK2 Method 
#---------------------------------
def RK2(f,t,y,h,N):
    for j in range(N):
        k1 = f(t, y)
        tilda = y + (2*h/3)*k1
        k2 = f(t + 2*h/3, tilda)
        y,t = y + (h/4)*k1 + (3*h/4)* k2, t + h
        print(j,y,t)
    return y
#--------------------------------
# Modified Euler Method 
#---------------------------------
def ModEuler(f,t,y,h,N):
    for j in range(N):
        k1 = f(t,y)
        tilda = y + (h/2)*k1
        k2 = f(t + h/2, tilda)
        y,t = y + h*k2, t + h
        print(j,y,t)
    return y
#----------------------------------


#--------TWO STEP ADAMS BASHFORTH with RK2 Method--------------------

def AB2(f,t,y,h,N): # 2 step Adams Bashforth 
    Y = np.zeros(N) # empty Y and T vectors
    T = np.zeros(N)
    Y[0], T[0] = y,t 
    
    # Optimal RK2 method -----------------
    
    k1 = f(t, y)
    tilda = y + (2*h/3)*k1
    k2 = f(t + 2*h/3, tilda)
    y,t = y + (h/4)*k1 + (3*h/4)* k2, t + h
    Y[1], T[1] = y, t
    
    # Calculates w_{1} and w{i+1}---------
        
    np.set_printoptions(precision=10)
        
    for i in range(1,N-1):
        Y[i+1] = (h/2) * ( 3*f(Y[i], T[i]) - f(Y[i-1], T[i-1]) ) + Y[i]
        T[i+1] = T[i] + h
        print(f(Y[i], T[i]), Y[i], T[i])

#--------------------------------------------------------------------
'''
def AB4(f,t,y,h,N):
    Y = np.zeros(N)
    T = np.zeros(N)
    Y[0], T[0] = y,t 
    
    # 4th order Runge Kutta Method calculate w1, w2, and w3
    n = 3
    for j in range(n): # calculate only w1,w2,w3
        k1 = f(t,y)
        k2 = f(t + h/2, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        y,t = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4), t + h
        Y[j],T[j] = y,t
        print(Y[j],T[j])
'''        


# Four step - Adams Bashforth Method 
# w_i+1 = w_i + h/24 * (55* f(ti, wi) - 59f(ti-1,wi-1) + 37f(ti-2,wi-2) - 9f(ti-3, wi-3))

def AB4(f,t,y,h,N):
    Y = np.zeros(N)
    T = np.zeros(N)
    Y[0], T[0] = y,t 
    
    # 4th order Runge Kutta Method calculate w1, w2, and w3
    n = 3 # first three evaluations 
    for j in range(n): # calculate w1,w2,w3
        k1 = f(t,y)
        k2 = f(t + h/2, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        y,t = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4), t + h
        Y[j],T[j] = y,t
        #print(Y[j], T[j]) # j[0,1,2] = [w1,w2,w3]
    print(Y[0])
    for i in range(n)