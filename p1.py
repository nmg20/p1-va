import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes
from skimage import data
import matplotlib.pyplot as plt

####
# Por consistencia se especifica que los arrays/matrices empleados
# deben ser de tipo np.array para poder extraer sus dimensiones
# siempre de la misma forma -> array.shape
# igualmente se tratan las imágenes como arrays de tipo np.array.
####


############################################################
# Dudas:
# EqualizeIntensity -> usar .cumsum()?
# Sobre filterImage -> eliminar padding para hacerlo más eficiente(?)
# SEs como arrays normales o como np.array?
# adjustIntensity en HighBoost  -> (dentro de la función?)
#                               -> tener en cuenta A en el rango de entrada?
# Erode con 0 en (0,0)
# Canny: -> en la umbralización usar 4 u 8 vecindad    
#   -> valores de umbrales para hacer testing
# Harris en general -> factor de sensitividad k?
#   -> usar gx y gy enteros o una vecindad para cada punto?
#   -> como incluir sigmaI?
#   -> return
# GaussKernel -> 
############################################################

# -> mirar morfologia con kernels destructivos
#     -> dilatacion con un kernel con un 0 en 0,0

# -> hacer pruebas en morfología con 8-conectividad
# -> probar casos de morfología especificando centro
# -> comprobar dilatación -> 0 en el origen

###################################

# Funciones auxiliares

def read_img(path):
  """
  Devuelve una imagen en punto flotante a partir de su path.
  return - array (imagen)
  """
  return cv.imread(path, cv.IMREAD_GRAYSCALE)/255.

def save_img(img, name):
  cv.imwrite(name, 255*img)

def show1(image):
  """
  Muestra la imagen proporcionada por pantalla.
  """
  height, width = image.shape  #Suponemos que ambas imágenes tienen el mismo tamaño (Original/Modificada)
  if (width>300):
    image = imutils.resize(image,width=300) #Resize proporcional sólo para mostrar las imágenes
  cv.imshow("", image)
  cv.waitKey(0)
  cv.destroyAllWindows()

def show(img1, img2):
  """
  Muestra dos imágenes una al lado de otra.
  """
  height, width = img1.shape  #Suponemos que ambas imágenes tienen el mismo tamaño (Original/Modificada)
  if (width>300):
    img1 = imutils.resize(img1,width=300) #Resize proporcional sólo para mostrar las imágenes
    img2 = imutils.resize(img2,width=300)
  # print("Height: ",height,"\tWidth: ",width)
  pack = np.concatenate((img1, img2), axis=1)
  cv.imshow("", pack)
  cv.waitKey(0)
  cv.destroyAllWindows()

def umbr(image, thres):
  """
  Umbraliza una imagen dado un threshold.
  """
  u = image.copy()
  u[u<thres]=0
  u[u>=thres]=1
  return u

##################################

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
  binsAux = np.linspace(np.min(inImage), np.max(inImage), nBins)
  outImage = np.zeros((inImage.shape),dtype='float32')
  for x in np.nditer(inImage):
    i = 0
    while x>binsAux[i]:
      i=i+1
    hist[i-1] = hist[i-1]+1
  cdf = hist.cumsum()
  # c_norm = cdf * hist.max() / cdf.max()
  c_norm = cdf - cdf[cdf>0].min() / cdf.max()
  outImage = np.interp(inImage, binsAux, c_norm)
  outImage = adjustIntensity(outImage,[],[0,1])
  return outImage

# Filtrado espacial: suavizado y realce

def filterImage(inImage, kernel): # [2]
  """
  Aplica un filtro mediante convolución de un kernel sobre una imagen.
  Antes de recorrer la imagen se le aplica un padding de tamaño p//2 y q//2,
  de esta forma la ventana siempre coge valores no nulos.
  - kernel = array/matriz de coeficientes (de tipo np.array -> para poder sacar .shape)
  """
  m, n = inImage.shape # Tamaño de la imagen
  p, q = kernel.shape # Tamaño del kernel
  a = p // 2
  b = q // 2
  outImage = np.zeros((m,n), dtype='float32') # Img resultado de menor tamaño
  padded = cv.copyMakeBorder(inImage,a,a,b,b,cv.BORDER_CONSTANT)
  for x in range(a, m+a, 1):
    for y in range(b, n+b, 1):
      window = padded[(x-a):(x+p-a),(y-b):(y+q-b)]
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
  # print("Size:",n,"\nCentro:",centro)
  kernel = np.zeros((1,n), dtype='float32')
  div = 1/math.sqrt(2*math.pi)*sigma
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
  m, n = inImage.shape
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
  m, n = inImage.shape
  p, q = SE.shape
  outImage = np.zeros((m,n), dtype='float32')
  padH, padV = math.floor(p/2), math.floor(q/2)
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
            # outImage[x-padH, y-padV]=min(window[i][j],SE[i][j])
  return outImage

def dilate(inImage, SE, center=[]):
  """
  Aplica el operador morfológico de dilatación.
    - inImage: imagen -> conjuntos de posiciones con 1s.
    - SE: elemento estructurante (con valores binarios).
    - center: origen del SE. Se asume que el [0, 0] es la esquina
        superior izquierda. Si está vacío, el centro es ([P/2]+1, [Q/2]+1).
  """
  m, n = inImage.shape
  p, q = SE.shape
  outImage = np.zeros((m,n), dtype='float32')
  padH, padV = math.floor(p/2), math.floor(q/2)
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
          # if window[i][j]==1 and SE[i][j]==1:
          if window[i][j]==1:
            outImage[x-padH, y-padV]=SE[i][j]
            # outImage[x-padH, y-padV]=min(window[i][j],SE[i][j])
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
    -Ac: ventana del complementario de la imagen original (tamaño del SE).
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
  a, b = math.floor(p/2), math.floor(q/2)
  if center == []:
    center=[a, b]
  outImage = cv.copyMakeBorder(inImage,a,a,b,b,cv.BORDER_CONSTANT, value=1)
  inverted = 1 - outImage
  toFill = set([])
  for (sx, sy) in seeds:
    # Se ajusta el valor de la seed con el padding
    toFill.add((sx+a,sy+b))
    while len(toFill)>0:
      sx,sy=toFill.pop()
      ar, ab, iz, de = sx-center[0], sx+p-center[0], sy-center[1], sy+q-center[1]
      # Se coge siempre una ventana del tamaño del SE
      window = outImage[ar:ab, iz:de]
      Ac = inverted[ar:ab, iz:de]
      outImage[ar:ab, iz:de] = fillWin(window, SE, Ac, sx, sy, toFill)
  return outImage[a:m+a,b:n+b]

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
  d = np.array([[0,1,0]])
  if operator == "Roberts":
    mx, my = np.array([[-1,0],[0,1]]), np.array([[0,-1],[1,0]])
  elif operator == "CentralDiff":
    # mx, my = a.T*d, d.T*a # Deberían ser el vector a y su transpuesta (no funciona x alguna razon)
    mx, my = a, a.T
  elif operator == "Prewitt":
    mx, my = b.T * a, a.T * b
  elif operator == "Sobel":
    mx, my = c.T * a, a.T*c
  return [filterImage(inImage, mx), filterImage(inImage, my)]

def direcciones(gx, gy):
  """
  Dadas dos matrices con los gradientes, registra para cada punto su dirección.
    -> arcotangente expresada en grados. -> multiplicada por 180 para usar grados convencionales.
    -> si son ángulos negativos se devuelve su simétrico (+180º)
  """
  m,n = gx.shape
  d = np.zeros((m,n), dtype='float32')
  for i in range(m):
    for j in range(n):
      d[i,j] = np.arctan2(gy[i,j],gx[i,j])*180/np.pi
      if d[i,j]<0:
        d[i,j] = d[i,j]+180
  return d

def mejora(img, sigma):
  """
  Devuelve una matriz con las direcciones y otra con las magnitudes.
  Aplica un suavizado gaussiano para reducir la influencia del ruido.
  """
  suavizado = gaussianFilter(img, sigma)
  gx, gy = gradientImage(suavizado, "Sobel")
  gtotal = np.sqrt((gx**2)+(gy**2))
  em = direcciones(gx, gy)
  return em, gtotal

def suprNoMax(mags, dirs):
  """
  Aplica supresión no máxima a los bordes de la imagen para obtener
  bordes de 1 pixel de grosor.
  """
  m,n = mags.shape
  sup = np.zeros((m,n), dtype='float32')
  for i in range(1,m-1):
    for j in range(1,n-1):
      # Vecinos para comparar máximos
      n1, n2 = 0,0
      # Horizontal
      if (0<dirs[i,j]<22.5 or 157.5<dirs[i,j]<180):
        n1, n2 = mags[i,j-1], mags[i,j+1]
      # Vertical
      elif (67.5<dirs[i,j]<112.5):
        n1, n2 = mags[i-1,j], mags[i+1,j]
      # Diagonal Ascendente
      elif (22.5<dirs[i,j]<67.5):
        n1, n2 = mags[i+1,j+1], mags[i-1,j-1]
      # Diagonal Descendente
      elif (112.5<dirs[i,j]<157.5):
        n1, n2 = mags[i+1,j-1], mags[i-1,j+1]
      # Comprobar si es máximo
      if (mags[i,j]>n1 and mags[i,j]>n2):
        sup[i,j]=mags[i,j]
  return sup

def umbralizacionConHisteresis(sup, tlow, thigh):
  """
  Se obtiene una imagen umbralizada a partir del resultado de aplicar
  la supresión no máxima.
  Se unen todos los bordes, juntando los débiles con los fuertes si están 4/8-conectados.
  """
  m, n = sup.shape
  umbralizado = np.zeros((m,n), dtype='float32')
  # Primero aplicamos una umbralización para diferenciar bordes fuertes de débiles.
  # Marcamos los bordes fuertes con un 1.0 y los débiles con 0.5 de intensidad.
  for i in range(m):
    for j in range(n):
      if (sup[i,j]>=thigh):
        umbralizado[i,j]=1
      elif (sup[i,j]>=tlow) and (sup[i,j]<thigh):
        umbralizado[i,j]=0.5
  # Aplicamos histéresis: nos quedamos con los bordes fuertes y los débiles
  # que estén unidos a otros fuertes
  # Se crea una copia del resultado de la umbralización con 
  # padding para no coger puntos fuera de la imagen
  umbralizado = cv.copyMakeBorder(umbralizado,1,1,1,1,cv.BORDER_CONSTANT,value=0.0)
  histeresis = np.zeros((m,n), dtype='float32')
  for i in range(m):
    for j in range(n):
      if (umbralizado[i,j]==1):
        histeresis[i,j] = 1
      elif (umbralizado[i,j]==0.5):
        if (umbralizado[i-1:i+2,j-1:j+1].sum()>0):
          histeresis[i,j] = 1
  return histeresis

def edgeCanny(inImage, sigma, tlow, thigh):
  """
  Algoritmo de detección de bordes Canny.
    - sigma: Parámetro del filtro gaussiano.
    - tlow, thigh: umbrales de histéresis bajo y alto.
  Proceso: 
    A)Mejora de la imagen.
      1. Suavizado gaussiano -> J(i,j)
      2. Para cada píxel  -> calcular gradientes -> Jx/Jy
                          -> calcular magnitud y orientación de los bordes Em(i,j)
    B)Supresión no máxima.
      1. Para cada píxel   -> encontrar dirección dk que aproxime Eo(i,j) (normal al borde)
      2. Asignar valor al nuevo píxel
          -> 0 si Es(i,j)<Es(n1) y Es(i,j)<Es(n2) (n1, n2 vecinos en dirección dk)
          -> Es(i,j) en otro caso
    C)Umbralización con histéresis.
      Para todo punto (i,j) en In
        1. Localizar In+1(i,j) no visitado, tal que In(i,j)>thigh
        2. A patir de In(i,j) recorrer píxeles conectados.
          -> marcar puntos recorridos como visitados
          -> ptos con magnitudes < thigh pero > tlow
            -> comprobar si están conectados a un máximo (> thigh)
  """ 
  direcciones, magnitudes = mejora(inImage,sigma)
  supresion = suprNoMax(magnitudes, direcciones)
  histeresis = umbralizacionConHisteresis(supresion,tlow, thigh)
  return umbralizado

# Operación opcional

def cornerHarris(inImage, sigmaD, sigmaI, t):
  """
  Operador Harris de detección de esquinas. 
    - sigmaD: escala de diferenciación.
    - sigmaI: escala de integración.
    - t: umbral de detección de esquinas.
    -> outCorners: mapa tras supresión no máxima y umbralización.
    -> harrisMap: cálculo de la métrica Harris para cada punto.
  Proceso:
    -> extraer Ix e Iy de la imagen (Sobel)
    -> aplicar una gaussiana con sigmaD a Ix2, Iy2 e Ix*Iy
    -> conformar matriz M -> obtener det(M) y aplicarle una gaussiana con sigmaI
    -> fórmula de harris (trace = gix2+giy2, establecer un k(?))
  """
  ix, iy = gradientImage(inImage,"Sobel")
  # show(adjustIntensity(ix,[],[0,1]),adjustIntensity(iy,[],[0,1]))
  ix2, iy2, ixy = ix**2, iy**2, ix*iy
  # show(ix2, iy2)
  # show1(adjustIntensity(ixy,[],[0,1]))
  gix2 = gaussianFilter(ix2,sigmaD)
  giy2 = gaussianFilter(iy2,sigmaD)
  gixiy = gaussianFilter(ixy,sigmaD)
  k = 0.05
  detA = gix2*giy2 - gixiy**2 # aplicar gaussiana con sigmaI(?)
  trace = gix2+giy2
  response = gaussianFilter(detA,sigmaI) - k*trace**2
  harrisMap = adjustIntensity(response,[],[0,1])
  # Versión que umbraliza la respuesta de Harris
  outCorners = umbr(1-harrisMap, t) #Se aplica una invesión para visualizarlo como en los ejemplos de clase
  # show(inImage,outCorners)
  # Versión con supresión no máxima y umbralización
  # sup = suprNoMax(np.sqrt((ix**2)+(iy**2)),direcciones(ix,iy))
  # outCorners = umbr(sup, t)
  return outCorners, harrisMap



def main():

  #
  #  Test de AdjustIntensity [v]
  #
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # image2 = adjustIntensity(image, [0, 1], [0.2, 0.6]) # Oscurece la imagen
  # image2 = adjustIntensity(image, [], [1, 0]) # Invierte la imagen
  # show(image,image2)

  #
  #  Test de EqualizeIntensity [~]
  #
  # image = read_img("./imagenes/eq.jpg")
  # image2 = equalizeIntensity(image, 256)
  # show(image,image2)

  #####

  #
  #  Test de FilterImage [v]
  #
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
  # kernel = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],
  #   [1,1,1,1,1],[1,1,1,1,1]])
  # kernel = np.array([[0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5],
  #   [0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5]])
  # kernel = np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0]]) # Blur Horizontal
  # kernel = np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]) # Blur vertical
  # image2 = filterImage(image, kernel)
  # show(image,adjustIntensity(image2,[],[0,1]))
  # show(image,image2)

  #
  # Test de GaussKernel1D [x]
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
  # Test de GaussianFilter
  #
  # image = read_img("./imagenes/circles1.png")
  # image = read_img("./imagenes/saltgirl.png")
  # image2 = gaussianFilter(image, 1.5)
  # show(image,adjustIntensity(image2,[],[0,1]))
  # show(image,image2)

  #
  # Test de MedianFilter [v]
  #
  # image = read_img("./imagenes/saltgirl.png")
  # image = read_img("./imagenes/lena.png")
  # image2 = medianFilter(image, 5)
  # image2 = medianFilter(image, 9) # Se difumina más
  # show(image, image2)

  #
  # Test de HighBoost [~]
  #
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/lena.png")
  # image = read_img("./imagenes/blur.jpg")
  # image2 = highBoost(image, 3, 'gaussian', 1)
  # image2 = highBoost(image, 1, 'gaussian', 1) # Laplaciano
  # image2 = highBoost(image, 1.5, 'median', 3)
  # show(image,adjustIntensity(image2,[],[0,1]))
  # show(image, image2)

  ##### Operadores Morfológicos

  # image = read_img("./imagenes/morphology/diagonal.png")
  # image = read_img("./imagenes/morphology/blob.png")
  # image = read_img("./imagenes/morphology/a34.png")
  # image = read_img("./imagenes/morphology/ex.png")
  # image = read_img("./imagenes/morphology/morph.png")

  #
  # Test de Erode
  #
  # SE = np.array([[1,1],[1,1]])
  # SE = np.array([[0,1,0],[0,1,0],[0,1,0]])
  # SE = np.array([[0,0,0],[1,1,1],[0,0,0]])
  # SE = np.array([[0,1,0],[1,1,1],[0,1,0]])
  # SE = np.array([[1,1,1],[1,1,1],[1,1,1]])
  # image2 = erode(image, SE, [])
  # show(image, image2)

  #
  # Test de Dilate
  #
  # SE = np.array([[0,1,0],[1,1,1],[0,1,0]])
  # SE = np.array([[0,1,0],[1,0,1],[0,1,0]])
  # SE = np.array([[1,1,1],[1,0,1],[1,1,1]])
  # image2 = dilate(image, SE, [])
  # show(image, image2)

  # Comprobar que (A-B)c = (Ac+B) 
  # Erosión y dilatación como operaiones complementarias
  # image = read_img("./imagenes/morphology/ex.png")
  # a = image.copy()
  # b = np.array([[0,1,0],[1,1,1],[0,1,0]])
  # r1 = 1 - (erode(a,b,[]))
  # r2 = dilate((1-a),b,[])
  # show(r1, r2)

  #
  # Test de Opening
  #
  # image = read_img("./imagenes/morphology/ex2.png")
  # SE = np.array([[0,1,0],[0,1,0],[0,1,0]])
  # SE = np.array([[1,1,1],[1,1,1],[1,1,1]])
  # SE = np.array([[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]])
  # image2 = opening(image, SE, [])
  # show(image, image2)

  #
  # Test de Closing
  #
  # SE = np.array([[0,1,0],[0,1,0],[0,1,0]])
  # SE = np.array([[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]])
  # image2 = closing(image, SE, [])
  # show(image, image2)

  #
  # Test de Fill
  #
  # image = read_img("./imagenes/morphology/closed.png")
  # image = read_img("./imagenes/morphology/closed8.png")
  # image = read_img("./imagenes/morphology/closed44.png")
  # image = read_img("./imagenes/morphology/closed10.png")
  # image = read_img("./imagenes/morphology/closed2.png")
  # seeds = [[1,1]]
  # seeds = [[5,5]]
  # seeds = [[2,2],[8,8]]
  # SE = np.array([[1,1,1],[1,1,1],[1,1,1]])
  # SE = np.array([[0,1,0],[1,1,1],[0,1,0]])
  # image2 = fill(image, seeds, [], [])
  # show(image, image2)

  #
  # Test de GradientImage
  #
  # image = read_img("./imagenes/morphology/closed.png")
  # image = read_img("./imagenes/grad7.png")
  # image = read_img("./imagenes/lena.png")
  # gx, gy = gradientImage(image, "Sobel")
  # gx = adjustIntensity(gx, [], [0,1])
  # gy = adjustIntensity(gy, [], [0,1])
  # show(image, gx)
  # show(image, gy)

  #
  # Test de EdgeCanny
  #
  # image = read_img("./imagenes/lena.png")
  # image2 = edgeCanny(image, 1.5, 0.2, 0.5)
  # show(image, image2)

  #
  # Test de CornersHarris
  #
  # image = read_img("./imagenes/grid.png")
  # cornersHarris, harrisMap = cornerHarris(image, 1.5, 0.5, 0.3)
  # show1(cornerHarris) # Errores en la función de mostrar imágenes (np.concatenate)
  # show(image,harrisMap)    # se decide mostrarlas separadamente
  # show1(cornerHarris)

if __name__ == "__main__":
  main()