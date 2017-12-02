import numpy as np
import math

def dense_gauss_kernel(sigma, x, y):

	# k = dense_gauss_kernel(sigma, x, y)
	#
	# Computes the kernel output for multi-dimensional feature maps x and y
	# using a Gaussian kernel with standard deviation sigma.

	#x in Fourier domain
	xf = np.fft.fft2(x)
	xx = x[:].T * x[:] #squared norm of x

	#general case, x and y are different
	yf = np.fft.fft2(y)
	yy = y[:].T * y[:]
	# take 2 equal x and y if need !!!!!
	#else
	    #auto-correlation of x, avoid repeating a few operations
	#    yf = xf;
	#    yy = xx;
	#end

	#cross-correlation term in Fourier domain
	xyf = xf * np.conj(yf)
	xy = np.sum(xyf, axis=2)
	xy = np.fft.ifft2(xy)
	xy = np.real(xy)	#to spatial domain

	#calculate gaussian response for all positions
	#for eazh element ??? !!!
	k = math.exp(-1 / sigma**2 * np.max(0, (xx + yy - 2 * xy) / x.size))

	return k