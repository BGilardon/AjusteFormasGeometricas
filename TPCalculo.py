import numpy as np
import random 
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))

# Ejercicio 1. Escribir un programa que implemente esta idea. Debe recibir una matriz de N
# pares (xi, yi) y devolver los parametros de la circunferencia.

def cuadradosMinimos(datos):
    A = np.ones((len(datos),3))
    for i in range(len(datos)):
        A[i] = [datos[i][0], datos[i][1], 1]
    At = A.T
    AtA = At@A
    b = np.ones(len(datos))
    for i in range(len(datos)):
        b[i] = (datos[i][0])**2 + (datos[i][1])**2
    Atb = At@b
    ConstantesABC = np.linalg.solve(AtA,Atb)
    x0 = ConstantesABC[0]/2
    y0 = ConstantesABC[1]/2
    r = (ConstantesABC[2] + x0**2 + y0**2)**(1/2)
    return (r,x0,y0) 

# Ejercicio 2. Para probar el programa podemos simular su aplicacion sobre datos generados
# artificialmente. Generar conjuntos de datos segun las siguientes pautas y graficarlos junto con
# el circulo obtenido a partir de ellos:
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

def setDatosCirculares(r,x0,y0,a=100,error=0,N=100):
    t = np.linspace(0, 2*np.pi*(a/100), N)
    XY = np.zeros((N,2))
    for i in range(N):
        XY[i,0] = (x0 + x0*random.uniform((-error),error)) + (r + r*random.uniform((-error),error))*np.cos(t[i])
        XY[i,1] = (y0 + y0*random.uniform((-error),error)) + (r + r*random.uniform((-error),error))*np.sin(t[i])
    return XY

# a) datos sobre un circulo completo, sin ruido,
# b) datos sobre un circulo completo, con ruido aleatorio
# c) datos sobre un arco de circunferencia de amplitud 0 < a ≤ 2π, con ruido.

a = setDatosCirculares(1,0,0, N=150)
b = setDatosCirculares(1,0,0,100,0.5,150)
b2 = setDatosCirculares(1,0,0,100,0.5,150)
c = setDatosCirculares(1,0,0,25,0.1,150)



def compararDatos(datos):
    plt.scatter(datos[:,0], datos[:,1], color='red', s=10)
    acm = cuadradosMinimos(datos)
    acmC = circulo(acm[0], acm[1], acm[2], len(datos))
    plt.plot(acmC[0], acmC[1], '--b')
    plt.scatter(acm[1], acm[2], marker='x', color='black')



# Ejercicio 3. Implementar un programa que reciba como input una funcion f y un punto z
# y calcule el vector gradiente de f evaluado en z, y un programa que calcule el hessiano de f
# evaluado en z. Ambos utilizando diferencias forward.


# Utilizo h = 10^(-8) para minimizar el error 
# (Si el h es mas chico, me epieza a dar datos erroneos por cancelaciones catastroficas)
def derivadaParcial(f,z,i,h=10**(-8)):  
    zi = z.copy()
    zi[i] += h
    Derivada = f(zi)/h - f(z)/h
    return Derivada

def gradiente(f, z):
    grad = np.zeros(len(z))
    for i in range(len(z)):
        grad[i] = derivadaParcial(f,z,i)
    return grad

def dobleDerivada(f,z,i,j,h=10**(-4)): #Como dentro de la doble derivada ha
    zj = z.copy()
    zj[j] = zj[j] + h
    derivadaDefEnij = derivadaParcial(f,zj,i)/h - derivadaParcial(f,z,i)/h
    return derivadaDefEnij

def hessiano(f,z):
    H = np.zeros((len(z),len(z)))
    for i in range(len(z)):
        for j in range(len(z)):
            H[i,j] = dobleDerivada(f,z,i,j)
    return H 



# Ejercicio 4. Implementar un programa que aplique el metodo de Newton, utilizando los
# programas del ejercicio anterior para computar el gradiente y el Hessiano y el m´etodo de
# Cholesky para resolver el sistema.

def vectorInicial(datos):
    v01 = np.sum(datos[:,0])/len(datos[:,0])
    v02 = np.sum(datos[:,1])/len(datos[:,1]) # (v01 , v02 ) es el punto promedio de los datos
    distancias = np.zeros(len(datos))
    for i in range(len(datos)):
        distancias[i] = ((datos[i,0]-v01)**2 + (datos[i,1]-v02)**2)**(1/2)
    v03 = np.sum(distancias)/len(distancias) # v03 es el promedio de las distancias a ese centro.
    
    return np.array([v01,v02,v03])

def matrizDeCholesky(A):
    n = len(A)
    L = np.zeros((n,n))
    
    L[0,0] = np.sqrt(A[0,0])
    
    for i in range(1,n):
        L[i,0] = A[i,0]/L[0,0]
    
    for j in range (1,n):
        L[j,j] = np.sqrt(A[j,j] - np.dot(L[j,0:j-1],L[j,0:j-1]))
        if j < n:
            for i in range(j+1, n):
                L[i,j] = (A[i,j] - np.dot(L[i,0:j-1],L[j,0:j-1]))/L[j,j]
    return L

def g(z):
    return z[0]*z[1]*z[0]
A = hessiano(g, [1,2,3])
L = matrizDeCholesky(A)
print(A, '\n','\n', L, '\n','\n', L.T, '\n','\n', np.dot(L, (L.T)))