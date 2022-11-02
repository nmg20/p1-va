import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes

# Dudas:
# - EqualizeIntensity -> usar numpy.hist o algo asi
# - GaussFilter1D -> 0 a N o -centro a centro?
# - HighBoost -> adjustIntensity con A?
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
  if (image.shape[1]>300):
    image = imutils.resize(image,width=300) #Resize proporcional sólo para mostrar las imágenes
  cv.imshow("", image)
  cv.waitKey(0)
  cv.destroyAllWindows()

def show2(img1, img2):
  """
  Muestra dos imágenes una al lado de otra.
  """
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

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]): # [2]
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
  # print("MinIn: ",min_in,"\tMaxIn: ",max_in,"\n\tMinOut: ",min_out,"\tMaxOut: ",max_out)
  return min_out + (((max_out - min_out)*inImage - min_in)/(max_in - min_in))

def equalizeIntensity(inImage, nBins=256):  # [0]
  """
  Ecualiza el histograma de la imagen.
  - nBins = número de bins empleados en el procesamiento.
  """
  m, n = np.shape(inImage)
  outImage = np.zeros((m,n), dtype='float32')
  histograma = np.zeros(nBins) # Array de zeros de tamaño nBins (se llena con los valores de las intensidades)
  bins = np.linspace(np.min(inImage), np.max(inImage), nBins) # Se crea un array que va desde la instensidad más baja a la mayor en nBins steps
  for x in np.nditer(inImage):
    i=0
    while x<=bins[i]:
      i=i+1
    histograma[i]=histograma[i]+1
  histAcum = histograma.cumsum()
  outImage = np.interp(inImage, bins[:-1], histAcum)

  return outImage

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
  outImage = np.zeros((m,n), dtype='float32') # Img resultado de menor tamaño
  padded = cv.copyMakeBorder(inImage,a,a,b,b,cv.BORDER_CONSTANT)
  for x in range(a, m+a, 1):
    for y in range(b, n+b, 1):
      window = padded[(x-a):(x+p-a),(y-b):(y+q-b)]
      # print("Window:\t(",x-a,"---",x+p-a,")\n\t(",y-b,"---",y+q-b)
      outImage[x-a,y-b] = (window * kernel).sum()
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
  # print("Kernel: ",kernel)
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

def medianFilter(inImage, filterSize):
  """
  Suaviza una imagen mediante un filtro de medianas bidimensional. 
  El tamaño del kernel del filtro viene dado por filterSize.
  """
  m, n = np.shape(inImage)
  outImage = np.zeros((m,n), dtype='float32')
  centro = filterSize//2
  for x in range(m):
    for y in range(n):
      limizq = max(0,y-centro)
      limder = min(n,y+filterSize-centro)
      limar = max(0,x-centro)
      limab = min(m,x+filterSize-centro)
      window = inImage[limar:limab,limizq:limder]
      outImage[x,y] = np.median(window)
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
  m, n = np.shape(inImage)
  realzado = np.zeros((m,n), dtype='float32')
  if method=='gaussian':
    suavizado = gaussianFilter(inImage, param)
  elif method=='median':
    suavizado = medianFilter(inImage, param)
  
  for x in range(0,m-1,1):
    for y in range(0,n-1,1):
        realzado[x,y] = A*inImage[x,y]-suavizado[x,y]
  # realzado = adjustIntensity(realzado, [], [0,1])
  return realzado
  # return adjustIntensity((A*inImage-suavizado),[],[0,1])

# Operadores morfológicos

def erode(inImage, SE, center=[]):
  """
  Aplica el operador morfológico de erosión.
    - inImage: imagen -> conjuntos de posiciones con 1s.
    - SE: elemento estructurante.
    - center: origen del SE. Se asume que el [0, 0] es la esquina
        superior izquierda. Si está vacío, el centro es ([P/2]+1, [Q/2]+1).
  """
  m, n = np.shape(inImage)
  p, q = np.shape(SE)
  outImage = inImage[(m,n), dtype='float32']

  for x in range(m):
    for y in range(n):
      window = inImage[(x-center[0]):(x+p-center[0]), (y-center[1]):(y+q-center[1])]
      # Comprobar si hay que erosionar
      eroded = false
      for eex in range(p):
        for eey in range(q):
          if window[eex,eey]==0 and SE[eex,eey]==1:
            eroded = true
            # outImage[(x-center[0]):(x+p-center[0]),
            #   (y-center[1]:(y+q-center[1]))] = 0.0
  return outImage

def dilate(inImage, SE, center=[]):
  """
  Aplica el operador morfológico de dilatación.
    - inImage: imagen -> conjuntos de posiciones con 1s.
    - SE: elemento estructurante.
    - center: origen del SE. Se asume que el [0, 0] es la esquina
        superior izquierda. Si está vacío, el centro es ([P/2]+1, [Q/2]+1).
  """
  return null

def opening(inImage, SE, center=[]):
  """
  Aplica el operador morfológico de apertura.
    - inImage: imagen -> conjuntos de posiciones con 1s.
    - SE: elemento estructurante.
    - center: origen del SE. Se asume que el [0, 0] es la esquina
        superior izquierda. Si está vacío, el centro es ([P/2]+1, [Q/2]+1).
  """
  return null

def closing(inImage, SE, center=[]):
  """
  Aplica el operador morfológico de cierre.
    - inImage: imagen -> conjuntos de posiciones con 1s.
    - SE: elemento estructurante.
    - center: origen del SE. Se asume que el [0, 0] es la esquina
        superior izquierda. Si está vacío, el centro es ([P/2]+1, [Q/2]+1).
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
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # image = read_img("./imagenes/77.png")
  # image = read_img("./imagenes/blob55.png")
  # image = read_img("./imagenes/point55.png")
  # image = read_img("./imagenes/x55.png")
  # image = read_img("./imagenes/salt77.png")
  # image = read_img("./imagenes/salt99.png")
  image = read_img("./imagenes/saltgirl.png")

  #
  #  Test de adjustIntensity
  #
  # image2 = adjustIntensity(image, [0, 0.5], [0, 1])
  # show2(image,image2)

  #
  #  Test de equalizeIntensity
  
  # image2 = equalizeIntensity(image, 256)
  show2(image,image2)

  #####

  #
  #  Test de filterImage
  #
  # kernel = [[0,1,0],[1,1,1],[0,1,0]]  # Aclara la imagen
  # kernel = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],
  #   [1,1,1,1,1],[1,1,1,1,1]]
  # kernel = [[0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5],
  #   [0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5]]
  # kernel = [[0,0.1,0],[0.1,0.1,0.1],[0,0.1,0]] # Oscurece la imagen
  # kernel = [[0,0.5,0],[0.5,0.5,0.5],[0,0.5,0]] # Aclara la imagen
  # image2 = filterImage(image, kernel)
  # show2(image,image2)

  #
  # Test de gaussKernel1D
  #
  # kernel1 = gaussKernel1D(0.5)
  # matrix = kernel1 * kernel1.T
  # print("Matriz: \n",matrix)
  # image2 = filterImage(image, matrix)
  # show2(image,image2)

  #
  # Test de gaussianFilter
  #
  # image2 = gaussianFilter(image, 1)
  # show2(image,image2)

  #
  # Test de medanFilter
  #
  # image2 = medianFilter(image, 5)
  # show2(image, image2)

  #
  # Test de medanFilter
  #
  # image2 = highBoost(image, 2, 'gaussian', 1)
  # image2 = highBoost(image, 2, 'median', 7)
  # show2(image, image2)

if __name__ == "__main__":
  main()