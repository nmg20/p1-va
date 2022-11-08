import argparse as argp
import p1

# funs = ["adj","eq","filt","gkern","gfilt","med","hboost",
#         "erode","dil","open","close","fill","grad","canny",
#         "harris"]
funsD = {"adj":"adjustIntensity","eq":"equalizeIntensity",
  "fil":"filterImage","gker":"gaussKernel1D", "gfil":"gaussianFilter",
  "med":"medianFilter", "hb":"highBooost","erd":"erode","dlt":"dilate",
  "op":"opening","cl":"closing","fl":"fill","grd":"gradientImage",
  "cny":"edgeCanny","hrs":"cornerHarris"}
imgs = ["eq","blur","circles","saltgirl","grid","grays","ex"]

def exec(fun, args):
  fun = getattr(p1, fun)
  return fun(args)

def main(args):
  if args:
    try:
      inImage = exec('read_img',args)
  else:

# PLAN:
# Se pueden probar las funciones individualmente o todas de una vez
# Se pueden probar con cada imagen individualmente o contra un pool
#   -> maybe pool específico para cada función(?)


if __name__ == "__main__":
  parser = argp.ArgumentParser(description=".")
  parser.add_argument('function', type=str, help="Función que se va a probar")