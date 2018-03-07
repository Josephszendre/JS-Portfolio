# image_segmentation.py
"""Volume 1A: Image Segmentation.
Logan Schelly
Math 321
10 November 2016
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg

# Problem 1: Implement this function.
def laplacian(A):
	'''
	Compute the Laplacian matrix of the adjacency matrix A.
	Inputs:
		A (array): adjacency matrix for undirected weighted graph,
			 shape (n,n)
	Returns:
		L (ndarray): Laplacian matrix of A
		  shape (n,n)
	'''
	D = np.diag(A.sum(axis=1)) #sum along the rows
	return D - A

# Problem 2: Implement this function.
def n_components(A,tol=1e-8):
	'''
	Compute the number of connected components in a graph
	and its algebraic connectivity, given its adjacency matrix.
	Inputs:
		A -- adjacency matrix for undirected weighted graph,
			 shape (n,n)
		tol -- tolerance value
	Returns:
		n_components -- the number of connected components
		lambda -- the algebraic connectivity
	'''
	if np.any(A < 0):
		raise ValueError("%r has negative entries."%A)
	
	#Get the eigenvalues of the laplacian of A
	eigenvalues = sp.linalg.eig(laplacian(A))[0]

	#cast them to reals
	eigenvalues = np.real(eigenvalues)
	
	#sort them
	eigenvalues.sort()
	
	#lambda is the second eigenvalue when the eigenvalues are in ascending order
	#and repeated according to multiplicity
	lambda_ = eigenvalues[1]
	
	#eigenvalues less than the tolerace are assumed to be 0
	if lambda_ < tol:
	    lambda_ = 0.
	
	#sum(eigenvalues>tol) gives the total number
	#of eigenvalues greater than the tolerance
	n_components = len(eigenvalues) - sum(eigenvalues>tol)
	
	return n_components, lamb_duh.real

# Problem 3: Implement this function.
def adjacency(filename="dream.png", radius = 5.0, sigma_I = .02, sigma_d = 3.0):
	'''
	Compute the weighted adjacency matrix for
	the image given the radius. Do all computations with sparse matrices.
	Also, return an array giving the main diagonal of the degree matrix.

	Inputs:
		filename (string): filename of the image for which the adjacency matrix will be calculated
		radius (float): maximum distance where the weight isn't 0
		sigma_I (float): some constant to help define the weight
		sigma_d (float): some constant to help define the weight
	Returns:
		W (sparse array(csc)): the weighted adjacency matrix of img_brightness,
			in sparse form.
		D (array): 1D array representing the main diagonal of the degree matrix.
	'''
	#Load image and convert to grayscale
	color_image, brightness_matrix = getImage(filename)
	
	#flatten image
	I = brightness_matrix.flatten() #Array of brightnesses
	
	#Initialize empty adjacency and degree matrices.
	M,N = brightness_matrix.shape
	W = sparse.lil_matrix((M*N,M*N), dtype = np.float64) #Adjacency array
	D = np.zeros(M*N) #Degree array
	
	#Fill in W
	for i in range(M*N):#iterate through each row
		#Find the pixels within r of this pixel
		#And find how far away they are
		indices, radii = getNeighbors(i, radius, N, M)
		
		#Calculate the weight to these neighbors and add it to W
		weights = np.exp(-abs(I[i] - I[indices])/sigma_I - radii/sigma_d) #9.1
		W[i, indices] = weights
		
		# ith entry of D is the ith row sum
		D[i] = sum(weights)
	
	#Convert W to csc format
	W = W.tocsc()
	
	return W,D

# Problem 4: Implement this function.
def segment(filename="dream.png"):
	'''
	Compute and return the two segments of the image as described in the text.
	Compute L, the laplacian matrix. Then compute D^(-1/2)LD^(-1/2),and find
	the eigenvector corresponding to the second smallest eigenvalue.
	Use this eigenvector to calculate a mask that will be usedto extract
	the segments of the image.
	Inputs:
		filename (string): filename of the image to be segmented
	Returns:
		seg1 (array): an array the same size as img_brightness, but with 0's
				for each pixel not included in the positive
				segment (which corresponds to the positive
				entries of the computed eigenvector)
		seg2 (array): an array the same size as img_brightness, but with 0's
				for each pixel not included in the negative
				segment.
	'''
	#Obtain the adjacency matrix and the diagonal matrix of its row sums
	W,D = adjacency(filename,radius = 5.0, sigma_I = .02, sigma_d = 3.0);
	
	#Define D^(-1/2) entrywise
	D_rcp_sqrt = D**(-1.0/2.0)
	
	#Use sparse matrices for faster computation
	D = sparse.spdiags(D, 0, D.size, D.size)
	D_rcp_sqrt = sparse.spdiags(D_rcp_sqrt, 0, D.size, D.size)
	
	# D - W is computed entrywise
	# Will do eigenvalue decomposition on the Laplacian to determine edges in picture
	L = D - W
	
	#Find second-smallest eigenvalue of D^(-1/2) * L * D^(-1/2)
	A = D_rcp_sqrt.dot( L.dot(D_rcp_sqrt))

	ss_eval, ss_evec = sparse.linalg.eigs(A, which= "SM")
	ss_eval = ss_eval[1]
	ss_evec = ss_evec[:,1]
	
	#Load image and convert to grayscale
	color_image, brightness_matrix = getImage(filename)
	
	#We need the dimensions of the image
	# explanation: shape is a 2-tuple, assigns m shape[0] and n shape[1] both integers
	m,n = brightness_matrix.shape	
	
	#Reshape eigenvector as an MxN array
	reshaped_evec = ss_evec.reshape((M,N))
	
	#Make into a mask by setting the positive entries to "True"
	#and the negative entries to "False"
	mask = reshaped_evec > 0

	#seg1 and seg2 must be the same size as image_brightness
	#negating normal mask entries, returns the negative values
	seg1 = brightness_matrix*mask
	seg2 = brightness_matrix*(~mask)
	
	return seg1, seg2

# Helper function used to convert the image into the correct format.
def getImage(filename='dream.png'):
	'''
	Reads an image and converts the image to a 2-D array of brightness
	values.

	Inputs:
		filename (str): filename of the image to be transformed.
	Returns:
		img_color (array): the image in array form
		img_brightness (array): the image array converted to an array of
			brightness values.
	'''
	img_color = plt.imread(filename)
	img_brightness = (img_color[:,:,0]+img_color[:,:,1]+img_color[:,:,2])/3.0
	return img_color,img_brightness
	

# Helper function for computing the adjacency matrix of an image
def getNeighbors(index, radius, height, width):
	'''
	Calculate the indices and distances of pixels within radius
	of the pixel at index, where the pixels are in a (height, width) shaped
	array. The returned indices are with respect to the flattened version of the
	array. This is a helper function for adjacency.

	Inputs:
		index (int): denotes the index in the flattened array of the pixel we are
				looking at
		radius (float): radius of the circular region centered at pixel (row, col)
		height, width (int,int): the height and width of the original image, in pixels
	Returns:
		indices (int): a flat array of indices of pixels that are within distance r
				   of the pixel at (row, col)
		distances (int): a flat array giving the respective distances from these
					 pixels to the center pixel.
	'''
	# Find appropriate row, column in unflattened image for flattened index
	row, col = index/width, index%width
	# Cast radius to an int (so we can use arange)
	r = int(radius)
	# Make a square grid of side length 2*r centered at index
	# (This is the sup-norm)
	x = np.arange(max(col - r, 0), min(col + r+1, width))
	y = np.arange(max(row - r, 0), min(row + r+1, height))
	X, Y = np.meshgrid(x, y)
	# Narrows down the desired indices using Euclidean norm
	# (i.e. cutting off corners of square to make circle)
	R = np.sqrt(((X-np.float(col))**2+(Y-np.float(row))**2))
	mask = (R<radius)
	# Return the indices of flattened array and corresponding distances
	return (X[mask] + Y[mask]*width, R[mask])

# Helper function used to display the images.
def displayPosNeg(img_color,pos,neg):
	'''
	Displays the original image along with the positive and negative
	segments of the image.

	Inputs:
		img_color (array): Original image
		pos (array): Positive segment of the original image
		neg (array): Negative segment of the original image
	Returns:
		Plots the original image along with the positive and negative
			segmentations.
	'''
	plt.subplot(131)
	plt.imshow(neg)
	plt.subplot(132)
	plt.imshow(pos)
	plt.subplot(133)
	plt.imshow(img_color)
	plt.show()

#--------------------------------
#Testing
#--------------------------------
if __name__ == "__main__":
	test_prob1 = True
	test_prob2 = False
	test_prob3 = True
	test_prob4 = True
	
	A = np.array(
	[[0,1,0,0,1,1]
	,[1,0,1,0,1,0]
	,[0,1,0,1,0,0]
	,[0,0,1,0,1,1]
	,[1,1,0,1,0,0]
	,[1,0,0,1,0,0]])
	
	B = np.array(
	[[0, 3, 0, 0, 0, 0]
	,[3, 0, 0, 0, 0, 0]
	,[0, 0, 0, 1, 0, 0]
	,[0, 0, 1, 0, 2,.5]
	,[0, 0, 0, 2, 0, 1]
	,[0, 0, 0,.5, 1, 0]])
	
	if test_prob1:
		print "9.1"
		print "The laplacian of this matrix:"
		print A
		print "Is this matrix:"
		print laplacian(A)
		print
		print "9.2"
		print "The laplacian of this matrix:"
		print B
		print "Is this matrix:"
		print laplacian(B)
	
	if test_prob2:
		print "9.1"
		print "For this adjacency matrix:"
		print A
		number_components, lamb_duh = n_components(A)
		print "Number of Components\t", number_components
		print "Algebraic Connectivity\t", lamb_duh
		print
		
		print "9.2"
		print "For this adjacency matrix:"
		print B
		number_components, lamb_duh = n_components(B)
		print "Number of Components\t", number_components
		print "Algebraic Connectivity\t", lamb_duh
		
	
	if test_prob3:
		A = getImage("Easy_Example.png")[1]
		print "Here is the brightness matrix"
		print A
		
		W,D = adjacency("Easy_Example.png", radius = 4)
		np.set_printoptions(precision=2)
		print W.toarray()
		W = W.toarray()
		print "These are the weights in the connecting to each pixel"
		for i in range(16):
			print W[i,:].reshape((4,4))
			print
		print D
		
	if test_prob4:
		picture = "dream.png"
		color_image, brightness = getImage(picture)
		seg1, seg2 = segment(picture)
		print "Problem 4"
		print "Positive of Image"
		print seg1
		print
		
		print "Negative of Image"
		print seg2
		print
		
		#Make things the right dimension to mask the color image
		#Nvm.  This was a misinterpretation of the assignment
		"""
		#however, this does make it show up in color.
		seg1 = np.array([seg1]*color_image.shape[2])
		seg2 = np.array([seg2]*color_image.shape[2])
		seg1 = seg1.swapaxes(0,1)
		seg2 = seg2.swapaxes(0,1)
		seg1 = seg1.swapaxes(1,2)
		seg2 = seg2.swapaxes(1,2)
		
		"""
		displayPosNeg(color_image, seg1, seg2)
		
		seg1 = seg1 != 0
		seg2 = seg2 != 0
		seg1 = np.array([seg1]*color_image.shape[2])
		seg2 = np.array([seg2]*color_image.shape[2])
		seg1 = seg1.swapaxes(0,1)
		seg2 = seg2.swapaxes(0,1)
		seg1 = seg1.swapaxes(1,2)
		seg2 = seg2.swapaxes(1,2)
		seg1 = color_image * seg1
		seg2 = color_image * seg2
		displayPosNeg(color_image, seg1, seg2)
		
	
