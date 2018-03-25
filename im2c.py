import math
import numpy as np

def im2c(im, w2c, color):
  # input im should be DOUBLE !
  # color=0 is color names out
  # color=-1 is colored image with color names out
  # color=1-11 is prob of colorname=color out;
  # color=-1 return probabilities
  # order of color names:
  #                black ,   blue   ,   brown       ,     grey       ,     green   ,  orange   ,   pink     ,   purple  ,  red     ,  white    , yellow
  color_values = [[0, 0, 0], [0, 0, 1], [0.5, 0.4, 0.25], [0.5, 0.5, 0.5], [0, 1, 0], [1, 0.8, 0], [1, 0.5, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]]

  #if nargin < 3:
  #   color = 0

  RR = im[:, :, 0]
  GG = im[:, :, 1]
  BB = im[:, :, 2]

  RR1 = np.zeros((RR.shape[0] * RR.shape[1], 1))
  GG1 = np.zeros((GG.shape[0] * GG.shape[1], 1))
  BB1 = np.zeros((BB.shape[0] * BB.shape[1], 1))

  #index_im = 1+floor(RR(:)/8)+32*floor(GG(:)/8)+32*32*floor(BB(:)/8);
  for i in range(im.shape[1]):
    for j in range(im.shape[0]):
      RR1[i * im.shape[0] + j][0] = math.floor(RR[j, i] / 8.0)
      GG1[i * im.shape[0] + j][0] = math.floor(GG[j, i] / 8.0)
      BB1[i * im.shape[0] + j][0] = math.floor(BB[j, i] / 8.0)
  #indexes: so need +1 ?? !!!! w2c from matlab, so it need ~~~

  index_im = 1 + RR1 + 32 * GG1 + 32 * 32 * BB1

  index_im = index_im.astype('int')
  index_im = index_im - 1

  if color == 0:
    w2cM = np.argmax(w2c, axis=1)
    w2cM = w2cM + 1
    w2cM = w2cM[index_im[:]]
    out = w2cM.reshape((im.shape[0], im.shape[1]), order='F')

  if (color > 0 and color < 12) or (color == -1):
    print("ERROR im2c")
    exit()

  if color == -2:
    w2cM = w2c[index_im[:]][:]
    out = w2cM.reshape((im.shape[0], im.shape[1], w2c.shape[1]), order='F')
  
  return out

  """
  if color >= 0 and color < 11:
    w2cM = w2c[:, color]
    w2cM = w2cM[index_im[:]]
    out = w2cM.reshape((im.shape[0], im.shape[1]))

  if color == -1:
    out = im
    w2cM = np.argmax(w2c, axis=1)
    w2cM = w2cM[index_im[:]]
    out2 = w2cM.reshape((im.shape[0], im.shape[1]))

    for i in range(im.shape[0]):
      for j in range(im.shape[1]):
        for c in range(3):
          out[i, j, c] = color_values[out2[i, j], c] * 255
  """
