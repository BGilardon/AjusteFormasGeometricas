import numpy as np
import random 
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))

def armarA(datos):
    A = np.ones((len(datos),3))

    for i in range(len(datos)):
        A[i] = [datos[i][0], datos[i][1], 1]
    
    return A

def armarb(datos):
    b = np.ones(len(datos))

    for i in range(len(datos)):
        b[i] = (datos[i][0])**2 + (datos[i][1])**2
    return b

def cuadradosMinimos(datos):
    A = armarA(datos)
    print(A)
    At = A.T
    AtA = At@A
    print(AtA)

    b = armarb(datos)
    Atb = At@b

    ABC = np.linalg.solve(AtA,Atb)
    
    x0 = ABC[0]/2
    y0 = ABC[1]/2
    r = (ABC[2] + x0**2 + y0**2)**(1/2)
    return (r,x0,y0) 


def circulo(r,x0,y0,N):
    t = np.linspace(0, 2*np.pi, N)
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        x[i] = x0 + r*np.cos(t[i])
        y[i] = y0 + r*np.sin(t[i])
    return (x,y)


#(r,x0,y0) son los datos del circulo, a va de 0 a 100 y representa que porcentaje del circulo se quiere graficar, 
# y el error es la cantidad de error relativo que se quiere aplicar
def setDatosCirculares(r,x0,y0,a,error,N):
    t = np.linspace(0, 2*np.pi*(a/100), N)
    XY = np.zeros((N,2))
    for i in range(N):
        XY[i,0] = (x0 + x0*random.uniform((-error),error)) + (r + r*random.uniform((-error),error))*np.cos(t[i])
        XY[i,1] = (y0 + y0*random.uniform((-error),error)) + (r + r*random.uniform((-error),error))*np.sin(t[i])
    return XY

'''a) datos sobre un circulo completo, sin ruido,
b) datos sobre un circulo completo, con ruido aleatorio
c) datos sobre un arco de circunferencia de amplitud 0 < a ≤ 2π, con ruido. ¿Que pasa
para a chico (del orden de π/4, por ejemplo)?'''

a = setDatosCirculares(1,0,0,100,0,40)
b = setDatosCirculares(1,0,0,100,0.1,150)
b2 = setDatosCirculares(1,0,0,100,0.8,150)
c = setDatosCirculares(1,0,0,25,0.1,150)

Sets = [a,b,b2,c]

def compararDatos(datos):
    plt.scatter(datos[:,0], datos[:,1], color='red', s=10)
    acm = cuadradosMinimos(datos)
    acmC = circulo(acm[0], acm[1], acm[2], len(datos))
    plt.plot(acmC[0], acmC[1], '--b')
    plt.scatter(acm[1], acm[2], marker='x', color='black')

    plt.show()

compararDatos(c)