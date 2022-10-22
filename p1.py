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
  # height, width = image.shape[:2]  #Suponemos que ambas imágenes tienen el mismo tamaño (Original/Modificada)
  if (image.shape[1]>300):
    image = imutils.resize(image,width=300) #Resize proporcional sólo para mostrar las imágenes
  cv.imshow("", image)
  cv.waitKey(0)
  cv.destroyAllWindows()

def show2(img1, img2):
  height, width = img1.shape[:2]  #Suponemos que ambas imágenes tienen el mismo tamaño (Original/Modificada)
  if (width>300):
    img1 = imutils.resize(img1,width=300) #Resize proporcional sólo para mostrar las imágenes
    img2 = imutils.resize(img2,width=300)
  # print("Height: ",height,"\tWidth: ",width)
  pack = np.concatenate((img1, img2), axis=1)
  cv.imshow("", pack)
  cv.waitKey(0)
  cv.destroyAllWindows()

# Histogramas: mejora de contraste

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]): # [1]
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

def equalizeIntensity(inImage, nBins=256):  # [0]
  """
  Ecualiza el histograma de la imagen.
  - nBins = número de bins empleados en el procesamiento.
  """
  return null

# Filtrado espacial: suavizado y realce

def filterImage(inImage, kernel): # [2]
  """
  Aplica un filtro mediante convolución de un kernel sobre una imagen.
  - kernel = array/matriz de coeficientes
  """
  m, n = np.shape(inImage) # Tamaño de la imagen
  p, q = np.shape(kernel) # Tamaño del kernel
  a = p // 2
  b = q // 2
  outImage = np.zeros((m-(a-1),n-(b-1)), dtype='float32') # Img resultado de menor tamaño
  # Como la convolucion da 0 en los bordes, se omiten
  for x in range(a, m-a, 1):
    for y in range(b, n-b, 1):
      window = inImage[(x-a):(x+p-a),(y-b):(y+q-b)]
      outImage[x,y] = (window * kernel).sum()/(np.sum(kernel))
    #   print(outImage[x,y], end="  ")
    # print("\n")
      
  return outImage

def gaussKernel1D(sigma): # [1]
  """
  Genera un array/matriz de una dimensión con una distribución
  gaussiana en base a la sigma dada.
  - sigma = desviación típica
  -> centro x=0 de la Gaussiana está en floor(N/2)+1
  -> N = 2*ceil(3*sigma)+1
  """
  n = 2 * math.ceil(3*sigma)+1
  # centro = math.floor(n//2)+1
  centro = math.floor(n//2)
  kernel = np.zeros((1,n), dtype='float32')
  # print("N: ",n,"\nCentro: ",centro,"\n\n")
  div = 1/math.sqrt(2*math.pi*sigma)
  exp = 2 * sigma**2
  # print("Div: ",div,"\n\n")
  # for x in range(0,n):
  #   kernel[0,x] = div * math.exp(-x**2/exp)
  for x in range(-centro,centro+1):
    kernel[0,x+centro] = div * math.exp(-x**2/exp)
  print("Kernel: ",kernel)
  return kernel

def gaussianFilter(inImage, sigma): # [1]
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
  kernel = gaussKernel1D(sigma)  
  matrix = kernel * kernel.T
  outImage = filterImage(inImage, matrix)
  return outImage

def medianFilter(inImage, filterSize): # [1]
  """
  Suaviza una imagen mediante un filtro de medianas bidimensional. 
  El tamaño del kernel del filtro viene dado por filterSize.
  """
  m, n = np.shape(inImage)
  outImage = inImage # Se usa la imagen original para evitar bordes en negro
  centro = filterSize//2
  for x in range(centro,m-centro,1): # Tomamos sólo las regiones del kernel que caen dentro de la imagen
    for y in range(centro,n-centro,1):
      window = inImage[(x-centro)::(x+centro),(y-centro)::(y+centro)]
      outImage[x,y] = np.sum(window)/(len(window)*len(window[0]))
  return outImage

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
  image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # image = read_img("./imagenes/77.png")
  # image = read_img("./imagenes/blob55.png")
  # image = read_img("./imagenes/point55.png")
  # image = read_img("./imagenes/x55.png")

  #
  #  Test de adjustIntensity
  #
  # show(image)
  # image2 = adjustIntensity(image, [0, 0.5], [0, 1])
  # show(image2)

  #####

  #
  #  Test de filterImage
  #
  # show(image)
  # kernel = [[0,1,0],[1,1,1],[0,1,0]]  # Aclara la imagen
  kernel = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],
    [1,1,1,1,1],[1,1,1,1,1]]
  # kernel = [[0,0.1,0],[0.1,0.1,0.1],[0,0.1,0]] # Oscurece la imagen
  # kernel = [[0,0.5,0],[0.5,0.5,0.5],[0,0.5,0]] # Aclara la imagen
  image2 = filterImage(image, kernel)
  # show(image2)
  show2(image,image2)

  #
  # Test de gaussKernel1D
  #
  # show(image)
  # kernel1 = gaussKernel1D(0.5)
  # matrix = kernel1 * kernel1.T
  # print("Matriz: \n",matrix)
  # show(image)
  # image2 = filterImage(image, matrix)
  # show(image2)

  #
  # Test de gaussianFilter
  #
  # show(image)
  # image2 = gaussianFilter(image, 0.5)
  # show(image2)

  #
  # Test de medanFilter
  #
  # show(image)
  # image2 = medianFilter(image, 4)
  # show(image2)


if __name__ == "__main__":
  main()