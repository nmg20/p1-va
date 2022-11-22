import argparse as argp
import sys
sys.path.append('../')
import p1

"""
Notación => N -> no está hecho lmao
            X -> pasa algo raro
            ~ -> no probado a fondo
            V -> ta piola
            S -> perfecto

Pruebas realizadas:
  -adjustIntensity S
  -equalizeIntensity ~
  -filterImage S
  -gaussKernel1D X
  -gaussianFilter V
  -medianFilter V
  -highBoost V
  -erode
  -dilate
  -opening
  -closing
  -fill
  -gradientImage
  -edgeCanny
  -cornerHarris N

"""




# funs = ["adj","eq","filt","gkern","gfilt","med","hboost",
#         "erode","dil","open","close","fill","grad","canny",
#         "harris"]
path = "./images/"
funsD = {"adj":"adjustIntensity","eq":"equalizeIntensity",
  "fltr":"filterImage","gker":"gaussKernel1D", 
  "gfil":"gaussianFilter","med":"medianFilter",
  "hb":"highBooost","erd":"erode","dlt":"dilate",
  "op":"opening","cl":"closing","fill":"fill",
  "grd":"gradientImage","can":"edgeCanny",
  "har":"cornerHarris"}
imgs = ["eq","blur","circles","circles1","saltgirl","grid",
  "grays","ex","morph","closed","closed2","lena"]

# kernels = [np.array([[0,1,0],[1,1,1],[0,1,0]]),
#   np.array([[0,0,0],[1,1,1],[0,0,0]]),
#   np.array([[0,1,0],[0,1,0],[0,1,0]]),
#   np.array([[1,1,1],[1,1,1],[1,1,1]]),]

# fun: [lista de listas de params]
# params = {"adj":[[[],[0,1]]],
#   "eq":[[],[256],[128]],
#   "fltr":[[kernels[0]],[kernels[1]],[kernels[2]]],
#   "gfil":[[1],[0.5],[1.4]],

#   }

rels = {"adj":["circles","circles1"], 
    "eq":["eq"], 
    "fltr":["circles","circles1","saltgirl"],
    "gfil":["circles","circles1","salgirl"],
    "med":["saltgirl"],
    "hb":["circles","saltgirl","blur"],
    "er":["ex","morph"],
    "dlt":["ex","morph"],
    "op":["ex","morph"],
    "cl":["ex","morph"],
    "fill":["closed", "closed2"],
    "grd":["lena"],
    "can":["lena","grays"],
    "har":["lena","grays","grid"]
  }

# def getImage(str):
#   return p1.read_image(path+str+".png")

def getImage(str):
  if str.endswith(".png"):
    return p1.read_image(str)
  else:
    return p1.read_image(path+str+".png")

def getImages():
  imgs = []
  for name in imgs:
    imgs.append(getImage(name))
  return imgs

def exec(fun, args):
  fun = getattr(p1, fun)
  return fun(args)

def main(args):
  print("heh")
  # if args:
  #   try:
  #     inImage = exec('read_img',args)
  # else:

# PLAN:
# Se pueden probar las funciones individualmente o todas de una vez
# Se pueden probar con cada imagen individualmente o contra un pool
#   -> maybe pool específico para cada función(?)

############## TESTS ##############

# AdjustIntensity
# -> probar a oscurecer/aclarar imágenes
# -> rangos invertidos deberían invertir la imagen tb

# EqualizeIntensity
# -> probar con varias imágenes -> usar adjustIntensity para modificarlas

# FilterImage
# -> Blur vertical/horizontal/diagonal(?)
# -> comprobar que elimina ruido impulsional
# -> checkear eficiencia -> padding

# GaussKernel1D
# -> probar que N está bien
# -> comprobar que genera bien los coeficientes

# GaussianFilter
# -> comprobar diferencia computacional entre vectores y matrices
# -> probar que elimina ruido impulsional
# -> proar para sigmas distintos

# MeanFilter
# -> probar que elimina ruido impulsional

# HighBoost
# -> probar con imágenes borrosas
# -> con A=1 -> Laplaciano
# -> algo con medianas (idk)

# Erode
# -> 0 en el origen
#

# Dilate
# -> 0 en el origen

# Opening/Closing
# -> usarlos como filtros de formas

# Fill
# -> bordes sin esquinas con vecindad 4 y 8 (SE)
# -> probar varios SEs
# -> probar varias seeds (muchas/desordenadas)

# GradientImage
# -> comprobar con todos los operadores
# -> ???

# EdgeCanny 
# -> probar con varios valores de threshold

# CornerHarris
#
#

###################################

if __name__ == "__main__":
  parser = argp.ArgumentParser(description=".")
  parser.add_argument('function', type=str, help="Función que se va a probar")
  args = parser.parse_args()
  main(args)


