import numpy as np
import cv2 as cv

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

# Histogramas: mejora de contraste

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):
  """
  Altera el rango dinámico de la imagen.
  - inRange = Rango de valores de intensidad de entrada.
  - outRange = Rango de valores de intensidad de salida.
  """
  if inRange == []:
    min_in = np.min(inImage)
    max_in = np.max(inImage)
  else :
    min_in = inRange[0]
    max_in = inRange[1]
  min_out = outRange[0]
  max_out = outRange[1]
  return min_out + (((max_out - min_out)*inImage - min_in)/(max_in - min_in))

def equalizeIntensity(inImage, nBins=256):
  """
  Ecualiza el histograma de la imagen.
  - nBins = número de bins empleados en el procesamiento.
  """
  return null

# Filtrado espacial: suavizado y realce

def filterImage(inImage, kernel):
  """
  Aplica un filtro mediante convolución de un kernel sobre una imagen.
  - kernel = array/matriz de coeficientes
  """
  m, n = np.shape(inImage) # Tamaño de la imagen
  p, q = np.shape(kernel) # Tamaño del kernel
  a = p // 2
  b = q // 2
  outImage = np.zeros((m-(a-1),n-(b-1)), dtype='float32') # Img resultado de menor tamaño
  for x in range(1, m-1, 1):
    for y in range(1, n-1, 1):
      window = inImage[(x-a):(x+p-a),(y-b):(y+q-b)]
      outImage[x,y] = (window * kernel).sum()
  return outImage

def gaussKernel1D(sigma):
  """
  Genera un array/matriz de una dimensión con una distribución
  gaussiana en base a la sigma dada.
  - sigma = desviación típica
  -> centro x=0 de la Gaussiana está en floor(N/2)+1
  -> N = 2*ceil(3*sigma)+1
  """
  n = 2 * math.ceil(3*sigma)+1
  centro = math.floor(n/2)+1
  kernel = np.zeros((1,n), dtype='float32')
  return null

def gaussianFilter(inImage, sigma):
  """
  Aplica un filtro de suavizado gaussiano sobre una imagen.
  El kernel del filtro viene dado por el sigma que genera la
  distribución de los coeficientes.
  Como el filtro es linealmente separable la matriz de coeficientes
  se puede generar multiplicando una matriz Nx1 por su matriz transpuesta.
  Pasos: 
    -> Multiplicar un kernel gaussiano 1xN por su matriz transpuesta
    -> Aplicar filterImage con la matriz anterior como kernel del filtro.
  """
  return null

def medianFilter(inImage, filterSize):
  """
  Suaviza una imagen mediante un filtro de medianas bidimensional. 
  El tamaño del kernel del filtro viene dado por filterSize.
  """
  return null

def highBoost(inImage, A, method, param):
  """
  Aplica el algoritmo de realce high boost.
  - A: factor de amplificación del filtro.
  - method: método de suavizado inicial
    - gaussian: filtro gaussiano
    - median: filtro de mediana
  - param: valor del parámetro del filtro de suavizado.
    -> sigma para gaussiano y size para medianas.
  """
  return null

# Operadores morfológicos

def erode(inImage, SE, center=[]):
  """

  """
  return null

def dilate(inImage, SE, center=[]):
  """

  """
  return null

def opening(inImage, SE, center=[]):
  """

  """
  return null

def closing(inImage, SE, center=[]):
  """

  """
  return null

def fill(inImage, seeds, SE=[], center=[]):
  """

  """
  return null

# Detección de bordes

def gradientImage(inImage, operator):
  """

  """
  return null

def edgeCanny(inImage, sigma, tlow, thigh):
  """

  """
  return null

# Operación opcional

def cornerHarris(inImage, sigmaD, sigmaI, t):
  """

  """
  return null



def main():
  #
  #  Test de adjustIntensity
  #
  image = read_img("./prueba/circles.png")
  show(image)
  image2 = adjustIntensity(image, [0, 0.5], [0, 1])
  show(image2)

  #####

  #
  #  Test de filterImage
  #
  show(image)
  kernel = [[0,1,0],[1,1,1],[0,1,0]]
  image2 = filterImage(image, kernel)
  show(image2)


if __name__ == "__main__":
  main()