import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes

# Funciones auxiliares

def read_img(path):
  return cv.imread(path, cv.IMREAD_GRAYSCALE)/255.

def show(image):
  cv.imshow("", image)
  cv.waitKey(0)
  cv.destroyAllWindows()

############# Funciones Auxiliares #############

def filterImage(inImage, kernel):
  m, n = np.shape(inImage)
  p, q = np.shape(kernel)
  a = p // 2
  b = q // 2
  outImage = np.zeros((m-(a-1),n-(b-1)), dtype='float32')
  for x in range(a, m-a, 1):
    for y in range(b, n-b, 1):
      window = inImage[(x-a):(x+p-a),(y-b):(y+q-b)]
      outImage[x,y] = (window * kernel).sum()/(np.sum(kernel))
    #   print(outImage[x,y], end="  ")
    # print("\n")
      
  return outImage

def gaussKernel1D(sigma):
  n = 2 * math.ceil(3*sigma)+1
  centro = math.floor(n//2)
  kernel = np.zeros((1,n), dtype='float32')
  div = 1/math.sqrt(2*math.pi*sigma)
  exp = 2 * sigma**2
  for x in range(-centro,centro+1):
    kernel[0,x+centro] = div * math.exp(-x**2/exp)
  # print("Kernel: ",kernel)
  return kernel

def gaussianFilter(inImage, sigma):
  kernel = gaussKernel1D(sigma)  
  matrix = kernel * kernel.T
  outImage = filterImage(inImage, matrix)
  return outImage

def medianFilter(inImage, filterSize):
  m, n = np.shape(inImage)
  outImage = inImage 
  centro = filterSize//2  
  for x in range(centro,m-centro,1):
    for y in range(centro,n-centro,1):
      window = inImage[(x-centro)::(x+centro),(y-centro)::(y+centro)]
      outImage[x,y] = np.sum(window)/(len(window)*len(window[0]))
  return outImage

############# Función Principal #############


def highBoost(inImage, A, method, param):
  m, n = np.shape(inImage)
  realzado = np.zeros((m,n), dtype='float32')
  if method=='gaussian':
    suavizado = gaussianFilter(inImage, param)
  elif method=='median':
    suavizado = medianFilter(inImage, param)
  
  for x in range(0,m-1,1):
    for y in range(0,n-1,1):
        realzado[x,y] = A*inImage[x,y]-suavizado[x,y]
  return realzado

#####

def main():
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # image = read_img("./imagenes/77.png")
  # image = read_img("./imagenes/blob55.png")
  # image = read_img("./imagenes/point55.png")
  # image = read_img("./imagenes/x55.png")
  image = read_img("./imagenes/salt99.png")


  # image2 = highBoost(image, 2, 'gaussian', 0.5)
  image2 = highBoost(image, 2, 'median', 3)
  show(image)
  show(image2)

if __name__ == "__main__":
  main()

