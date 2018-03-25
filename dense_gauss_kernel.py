import numpy as np
import math

def dense_gauss_kernel(sigma, x, y):

	# k = dense_gauss_kernel(sigma, x, y)
	#
	# Computes the kernel output for multi-dimensional feature maps x and y
	# using a Gaussian kernel with standard deviation sigma.

	#x in Fourier domain
	xf = np.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype='complex')
	for i in range(x.shape[2]):
		xf[:,:,i] = np.fft.fft2(x[:,:,i])
	x_dots = x.reshape((x.shape[0] * x.shape[1] * x.shape[2], 1), order='F')
	xx = np.dot(x_dots.T, x_dots) #squared norm of x

	#general case, x and y are different
	yf = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype='complex')
	for i in range(y.shape[2]):
		yf[:,:,i] = np.fft.fft2(y[:,:,i])
	y_dots = y.reshape((y.shape[0] * y.shape[1] * y.shape[2], 1), order='F')
	yy = np.dot(y_dots.T, y_dots)
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
	tmp = -1 / sigma**2 * np.maximum(0, (xx + yy - 2 * xy) / x.size)
	k = np.exp(tmp)

	return k

def dense_gauss_kernel_2(sigma, x):

	# k = dense_gauss_kernel(sigma, x, y)
	#
	# Computes the kernel output for multi-dimensional feature maps x and y
	# using a Gaussian kernel with standard deviation sigma.

	#x in Fourier domain
	xf = np.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype='complex')
	for i in range(x.shape[2]):
		xf[:,:,i] = np.fft.fft2(x[:,:,i])
	x_dots = x.reshape((x.shape[0] * x.shape[1] * x.shape[2], 1), order='F')
	xx = np.dot(x_dots.T, x_dots) #squared norm of x

	#general case, x and y are different
	#yf = np.fft.fft2(y)
	#yy = y[:].T * y[:]
	# take 2 equal x and y if need !!!!!
	#else
	#auto-correlation of x, avoid repeating a few operations
	yf = xf
	yy = xx
	#end

	#cross-correlation term in Fourier domain
	xyf = xf * np.conj(yf)
	xy = np.sum(xyf, axis=2)
	xy = np.fft.ifft2(xy)
	xy = np.real(xy)	#to spatial domain

	#calculate gaussian response for all positions
	tmp = -1 / sigma**2 * np.maximum(0, (xx + yy - 2 * xy) / x.size)
	k = np.exp(tmp)

	return k