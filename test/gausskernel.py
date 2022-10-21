import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes

# Dudas:
# - EqualizeIntensity -> usar numpy.hist o algo asi
# - FilterImage -> Omitir bordes o añadir padding?
# - GaussFilter1D -> 0 a N o -centro a centro?
#

# Funciones auxiliares

def read_img(path):
  """
  Devuelve una imagen en punto flotante a partir de su path.
  return - array (imagen)
  """
  return cv.imread(path, cv.IMREAD_GRAYSCALE)/255.

def show(image):
  """
  Muestra la imagen proporcionada por pantalla.
  """
  cv.imshow("", image)
  cv.waitKey(0)
  cv.destroyAllWindows()

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
  centro = math.floor(n//2)
  kernel = np.zeros((1,n), dtype='float32')
  div = 1/math.sqrt(2*math.pi*sigma)
  for x in range(-centro,centro+1):
    a = div * math.exp(-x**2/(2 * sigma**2))
    kernel[0,x+centro] = float("{0:.4f}".format(a))
  kernel = kernel * kernel.T
  print("Kernel: \n\t",kernel)
  # return kernel

def alt_gaussKernel1D(size, sigma, mu):
  x, y = np.meshgrid(np.linspace(-2, 2, size),
                    np.linspace(-2, 2, size))
  dst = np.sqrt(x**2+y**2)
  normal = 1/(2.0*np.pi*sigma**2)
  gauss = np.exp(-((dst-mu)**2/(2.0*sigma**2)))*normal
  print("OG Kernel\n\t",gauss)
  # return gauss

#####

def main():
  sigma=0.5
  n = 2 * math.ceil(3*sigma)+1
  gaussKernel1D(0.5)
  alt_gaussKernel1D(n,sigma,0)
  print("\n#############################################")
  sigma=1.0
  n = 2 * math.ceil(3*sigma)+1
  gaussKernel1D(1)
  alt_gaussKernel1D(n,sigma,0)
  print("\n")
  sigma=0.15
  n = 2 * math.ceil(3*sigma)+1
  gaussKernel1D(0.15)
  alt_gaussKernel1D(n,sigma,0)
  print("\n")
  

if __name__ == "__main__":
  main()

