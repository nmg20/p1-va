import math
import numpy as np
import cv2 as cv
import sys
sys.path.append('../')
import p1

# Dudas:
# - EqualizeIntensity -> usar numpy.hist o algo asi
# - FilterImage -> Omitir bordes o añadir padding?
# - GaussFilter1D -> 0 a N o -centro a centro?
#

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

# def alt_gaussKernel1D(size, sigma, mu):
#   x, y = np.meshgrid(np.linspace(-2, 2, size),
#                     np.linspace(-2, 2, size))
#   dst = np.sqrt(x**2+y**2)
#   normal = 1/(2.0*np.pi*sigma**2)
#   gauss = np.exp(-((dst-mu)**2/(2.0*sigma**2)))*normal
#   print("OG Kernel\n\t",gauss)
#   # return gauss

# def gaussKernel1D(sigma):
#     size = round(2*(3*sigma)+1)
#     kernel = np.zeros(size)
#     mid = math.floor(size/2)
#     print("Size:", size,"\nMid:",mid)
#     kernel=[(1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]
#     return np.array(kernel)

def gaussKernel1D(sigma):
  n = 2 * int(math.ceil(3*sigma)) + 1
  kernel = np.zeros((1,n), dtype='float32')
  center = n//2
  den = math.sqrt(2 * math.pi) * sigma
  denExp = 2 * (sigma**2)
  for x in range(-center, center+1):
    kernel[0, x + center] = math.exp( -x**2 / denExp ) / den
  return kernel


#####

def main():
  kernel = gaussKernel1D(0.5)
  k = kernel * kernel.T
  print(k)
  

if __name__ == "__main__":
  main()

