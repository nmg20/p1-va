import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes
from skimage import data

############################################################
# Dudas:
# - HighBoost -> adjustIntensity con A?
# A COMO UMBRAL DE ENTRADA
# - Añadir padding para los operadores morfológicos    
# NO NECESARIO        
#   -> en erode funciona(?) -> saber si hay casos en los que pueda fallar
# Fill debería funcionar en fondos/ figuras cortadas por los bordes de la imagen?
# Gradient -> Operador de CentralDiff 
#     -> [-1,0,1].T*[-1,0,1]?
############################################################


# Funciones auxiliares

def read_img(path):
  """
  Devuelve una imagen en punto flotante a partir de su path.
  return - array (imagen)
  """
  return cv.imread(path, cv.IMREAD_GRAYSCALE)/255.

def save_img(img, name):
  cv.imwrite(name, 255*img)

def show(image):
  """
  Muestra la imagen proporcionada por pantalla.
  """
  if (image.shape[1]>300):
    image = imutils.resize(image,width=300) #Resize proporcional sólo para mostrar las imágenes
  cv.imshow("", image)
  cv.waitKey(0)
  cv.destroyAllWindows()

def show(img1, img2):
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

def equalizeIntensity(inImage, nBins=256):  # [1]
  """
  Ecualiza el histograma de la imagen.
  - nBins = número de bins empleados en el procesamiento.
  """
  hist = np.zeros(nBins, dtype='int')
  binsAux = np.linspace(np.min(inImage), np.max(inImage), nBins+1)
  outImage = np.zeros((np.shape(inImage)),dtype='float32')
  for x in np.nditer(inImage):
    i = 0
    while x>binsAux[i]:
      i=i+1
    hist[i-1] = hist[i-1]+1
  cdf = hist.cumsum()
  c_norm = cdf * hist.max() / cdf.max()
  outImage = np.interp(inImage, binsAux[:-1], c_norm)
  outImage = adjustIntensity(outImage,[],[0,1])
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
  --> devuelve un array (0,n) -> importante para hacer la transposición
  """
  n = 2 * math.ceil(3*sigma)+1
  centro = math.floor(n/2)
  kernel = np.zeros((1,n), dtype='float32')
  div = 1/math.sqrt(2*math.pi*sigma)
  exp = 2 * sigma**2
  for x in range(-centro,centro+1):
    kernel[0,x+centro] = div * math.exp(-x**2/exp)
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
  # matrix = kernel * kernel.T
  # outImage = filterImage(inImage, matrix)
  outImage = filterImage(filterImage(inImage, kernel), kernel.T)
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
      # Sólo se consideran los valores de la ventana situados dentro de la imagen
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
  
  # for x in range(m):
  #   for y in range(n):
  #       realzado[x,y] = A*inImage[x,y]-suavizado[x,y]
  # realzado = adjustIntensity(realzado, [], [0,1])
  realzado = adjustIntensity(A*inImage-suavizado, [], [0,1])
  # return A*inImage-suavizado
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
  outImage = np.zeros((m,n), dtype='float32')
  padH, padV = p//2, q//2
  if center==[]:
    center=[padH, padV]
  pad = cv.copyMakeBorder(inImage, padH, padH, padV, padV, cv.BORDER_CONSTANT)
  for x in range(padH, m+padH, 1): #Recorremos la distancia de la imagen
    for y in range(padV, n+padV, 1): # original dentro de la paddeada
      limar = x-center[0]
      limab = x+p-center[0]
      limizq = y-center[1]
      limder = y+q-center[1]
      window = pad[limar:limab, limizq:limder]
      outImage[x-padH,y-padV]=inImage[x-padH, y-padV]
      # Se comprueba que se erosiona el píxel
      for i in range(p):
        for j in range(q):
          if window[i][j]==0 and SE[i][j]==1:
            outImage[x-padH, y-padV]=0
  return outImage

def dilate(inImage, SE, center=[]):
  """
  Aplica el operador morfológico de dilatación.
    - inImage: imagen -> conjuntos de posiciones con 1s.
    - SE: elemento estructurante.
    - center: origen del SE. Se asume que el [0, 0] es la esquina
        superior izquierda. Si está vacío, el centro es ([P/2]+1, [Q/2]+1).
  """
  m, n = np.shape(inImage)
  p, q = np.shape(SE)
  outImage = np.zeros((m,n), dtype='float32')
  padH, padV = p//2, q//2
  if center==[]:
    center=[padH, padV]
  pad = cv.copyMakeBorder(inImage, padH, padH, padV, padV, cv.BORDER_CONSTANT)
  for x in range(padH, m+padH, 1): #Recorremos la distancia de la imagen
    for y in range(padV, n+padV, 1): # original dentro de la paddeada
      limar = x-center[0]
      limab = x+p-center[0]
      limizq = y-center[1]
      limder = y+q-center[1]
      window = pad[limar:limab, limizq:limder]
      outImage[x-padH,y-padV]=inImage[x-padH, y-padV]
      # Se comprueba que se dilata la región en torno al píxel
      for i in range(p):
        for j in range(q):
          if window[i][j]==1 and SE[i][j]==1:
            outImage[x-padH, y-padV]=1
  return outImage

def opening(inImage, SE, center=[]):
  """
  Aplica el operador morfológico de apertura.
    - inImage: imagen -> conjuntos de posiciones con 1s.
    - SE: elemento estructurante.
    - center: origen del SE. Se asume que el [0, 0] es la esquina
        superior izquierda. Si está vacío, el centro es ([P/2]+1, [Q/2]+1).
  """
  return dilate(erode(inImage,SE,center),SE,center)

def closing(inImage, SE, center=[]):
  """
  Aplica el operador morfológico de cierre.
    - inImage: imagen -> conjuntos de posiciones con 1s.
    - SE: elemento estructurante.
    - center: origen del SE. Se asume que el [0, 0] es la esquina
        superior izquierda. Si está vacío, el centro es ([P/2]+1, [Q/2]+1).
  """
  return erode(dilate(inImage, SE, center), SE, center)

def fillWin(window, SE, Ac, x, y, filled):
  """
  Función auxiliar que llena una región delimitada en una ventana (window), 
  usando SE y el invertido del fondo (Ac) para hacer la dilatación condicional.
    -window: ventana de la imagen a rellenar (tamaño del SE).
    -Ac: ventana del invertido de la imagen original (tamaño del SE).
    -x, y: coordenadas globales del centro del SE en la imgen.
    -filled: set con las coordenadas de los puntos que se rellenan.
  -> devuelve la matriz de la región rellenada.
  """
  m, n = window.shape
  result = window.copy()
  for i in range(m):
    for j in range(n):
      if ((window[i][j]==0 and SE[i][j]==1) and Ac[i][j]==1):
        filled.add((i+x-1,j+y-1))
        result[i][j]=1
  return result

def fill(inImage, seeds, SE=[], center=[]):
  """
  Realiza el llenado morfológico de una o varias regiones.
    - seeds: matriz Nx2 con N coordenadas (fila, columna) de los
      puntos por donde se empieza a llenar la imagen.
    - SE: matriz PxQ binaria que define el elemento estructurante
      de conectividad. Si es vacío se supone conectividad 4.
  """
  if SE == []:
    # Por defecto se considera conectividad-4
    SE = np.array([[0,1,0],[1,1,1],[0,1,0]])
  m, n = inImage.shape
  p, q = SE.shape
  if center == []:
    center=[p//2,q//2]
  outImage = inImage.copy() # Copia de la imagen sobre la que trabajamos
  inverted = 1-np.asarray(inImage) # Imagen invertida -> dilatación condicional
  # show(inImage, inverted)
  filled = set([])
  for (sx, sy) in seeds:
    filled.add((sx,sy))
    while len(filled)>0:
      (sx,sy)=filled.pop()
      limar = max(0,sx-center[0])
      limab = min(m,sx+p-center[0])
      limizq = max(0,sy-center[1])
      limder = min(n,sy+q-center[1])
      window = outImage[limar:limab, limizq:limder]
      Ac = inverted[limar:limab, limizq:limder]
      outImage[limar:limab, limizq:limder] = fillWin(window, SE, Ac, sx, sy, filled)
  return outImage

# Detección de bordes

def gradientImage(inImage, operator):
  """
  Devuelve las componentes Gx y Gy del gradiente de una imagen.
    - operator: define el método que se usa para obtener el gradiente.
      -->Roberts, CentralDiff(de Prewitt/Sobel), Prewitt, Sobel.
  Según el operador escoger una matriz a modo de SE
    -> aplicarla convolucionalmente por la imagen
  """
  m, n = inImage.shape
  gx, gy = np.zeros((m,n), dtype='float32'), np.zeros((m,n), dtype='float32')
  a = np.array([[-1,0,1]])
  b = np.array([[1,1,1]])
  c = np.array([[1,2,1]])
  d = np.array([[1,0,1]])
  if operator == "Roberts":
    mx, my = np.array([[-1,0],[0,1]]), np.array([[0,-1],[1,0]])
  elif operator == "CentralDiff":
    mx, my = a.T*a, a.T*a
  elif operator == "Prewitt":
    mx, my = b.T * a, a.T * b
  elif operator == "sSobel":
    mx, my = c.T * a, a.T*c
  p, q = mx.shape # Se toman las dimensiones de una matriz indistinta de las dos para hacer padding
  a, b = p//2, q//2
  pad = cv.copyMakeBorder(inImage,a,a,b,b,cv.BORDER_CONSTANT)
  for x in range(a,m+a,1):
    for y in range(b,n+b,1):
      # Cogemos la ventana de la imagen paddeada
      window = pad[(x-a):(x+p-a),(y-b):(y+q-b)]
      gx[x-a,y-b] = (window*mx).sum() # Cambiar por vectores para hacer más eficiente
      gy[x-a,y-b] = (window*my).sum() # -> matrices linealmente separables
  return [gx, gy]

def edgeCanny(inImage, sigma, tlow, thigh):
  """
  Algoritmo de detección de bordes Canny.
    - sigma: Parámetro del filtro gaussiano.
    - tlow, thigh: umbrales de histéresis bajo y alto.
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
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # image2 = adjustIntensity(image, [0, 0.5], [0, 1])
  # show(image,image2)

  #
  #  Test de equalizeIntensity
  
  # image = read_img("./imagenes/eq.jpg")
  # image2 = equalizeIntensity(image, 256)
  # show(image,image2)

  #####

  #
  #  Test de filterImage
  #
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # kernel = [[0,1,0],[1,1,1],[0,1,0]]  # Aclara la imagen
  # kernel = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],
  #   [1,1,1,1,1],[1,1,1,1,1]]
  # kernel = [[0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5],
  #   [0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5]]
  # kernel = [[0,0.1,0],[0.1,0.1,0.1],[0,0.1,0]] # Oscurece la imagen
  # kernel = [[0,0.5,0],[0.5,0.5,0.5],[0,0.5,0]] # Aclara la imagen
  # image2 = filterImage(image, kernel)
  # show(image,image2)

  #
  # Test de gaussKernel1D
  #
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # image = read_img("./imagenes/saltgirl.png")
  kernel1 = gaussKernel1D(0.5)
  # matrix = kernel1 * kernel1.T
  # print("Matriz: \n",matrix)
  # image2 = filterImage(image, matrix)
  # show(image,image2)

  #
  # Test de gaussianFilter
  #
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # image = read_img("./imagenes/saltgirl.png")
  # image2 = gaussianFilter(image, 1)
  # show(image,image2)

  #
  # Test de medanFilter
  #
  # image = read_img("./imagenes/saltgirl.png")
  # image2 = medianFilter(image, 5)
  # show(image, image2)

  #
  # Test de highBoost
  #
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/saltgirl.png")
  # image = read_img("./test/pruebas/blur.png")
  # image2 = highBoost(image, 3, 'gaussian', 1)
  # image2 = highBoost(image, 2, 'median', 3)
  # show(image, image2)

  ##### Operadores Morfológicos

  # image = read_img("./imagenes/morphology/diagonal.png")
  # image = read_img("./imagenes/morphology/blob.png")
  # image = read_img("./imagenes/morphology/a34.png")
  # image = read_img("./imagenes/morphology/ex.png")

  #
  # Test de Erode
  #
  # SE = [[1,1],[1,1]]
  # image2 = erode(image, SE, [])
  # show(image, image2)

  #
  # Test de Dilate
  #
  # SE = [[0,1,0],[1,1,1],[0,1,0]]
  # image2 = dilate(image, SE, [])
  # show(image, image2)

  #
  # Test de Opening
  #
  # SE = [[0,1,0],[0,1,0],[0,1,0]]
  # SE = [[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]]
  # image2 = opening(image, SE, [])
  # show(image, image2)

  #
  # Test de Closing
  #
  # SE = [[0,1,0],[0,1,0],[0,1,0]]
  # SE = [[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]]
  # image2 = closing(image, SE, [])
  # show(image, image2)

  #
  # Test de Fill
  #
  # image = read_img("./imagenes/morphology/closed.png")
  # image = read_img("./imagenes/morphology/closed44.png")
  # image = read_img("./imagenes/morphology/closed10.png")
  # image = read_img("./imagenes/morphology/closed2.png")
  # seeds = [[1,1]]
  # seeds = [[5,5]]
  # seeds = [[2,2],[8,8]]
  # image2 = fill(image, seeds, [], [])
  # show(image, image2)

  #
  # Test de GradientImage
  #
  # image = read_img("./imagenes/morphology/closed.png")
  # image = read_img("./imagenes/grad7.png")
  # image = read_img("./imagenes/lena.png")
  # gx, gy = gradientImage(image, "roberts")
  # gx = adjustIntensity(gx, [], [0,1])
  # gy = adjustIntensity(gy, [], [0,1])
  # show(image, gx)
  # show(image, gy)

if __name__ == "__main__":
  main()