import math
import numpy as np
import cv2 as cv
import sys
sys.path.append('../')
import p1

def gaussKernel1D(sigma): # [1]
  """
  Genera un array/matriz de una dimensión con una distribución
  gaussiana en base a la sigma dada.
  - sigma = desviación típica
  -> centro x=0 de la Gaussiana está en floor(N/2)+1
  -> N = 2*ceil(3*sigma)+1
  """
  n = 2 * math.ceil(3*sigma)+1
  centro = math.floor(n/2)
  print("Size:",n,"\nCentro:",centro)
  kernel = np.zeros((1,n), dtype='float32')
  div = 1/math.sqrt(2*math.pi)*sigma
  exp = 2 * sigma**2
  for x in range(-centro,centro+1):
    kernel[0,x+centro] = div * math.exp(-x**2/exp)
  return kernel

# def gaussKernel1D(sigma):
#     size = round(2*(3*sigma)+1)
#     kernel = np.zeros(size)
#     mid = math.floor(size/2)
#     kernel=[(1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]
#     return kernel

#####

def main():
  kernel = gaussKernel1D(0.5)
  k = kernel * kernel.T
  print(k)
  

if __name__ == "__main__":
  main()

