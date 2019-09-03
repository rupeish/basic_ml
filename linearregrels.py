#imports
import numpy as np
import matplotlib.pyplot as plt

def est_coeff(x,y):
	m=np.size(x); #number of trainingexamples

	mean_x,mean_y =  np.mean(x),np.mean(y) #mean

	ss_xy = np.sum(y*x) - m*mean_y*mean_x #cross deviation
	ss_xx = np.sum(x*x) - m*mean_x*mean_x

	b1 = ss_xy/ss_xx #regression coefficient
	b0 = mean_y - b1*mean_x

	return(b0,b1)

def plot_regressionline(x,y,b):
	plt.scatter(x, y) #original data
 	
	y= b[0] + b[1]*x #predicted data

	plt.plot(x, y, color = "g") #regression line
 
	plt.xlabel('x') #labels
	plt.ylabel('y') 
 
	plt.show()
def main(): 
	 
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  #training data
	y = np.array([2, 3, 5, 6, 8, 9, 12, 14, 15, 16]) 

	b = est_coeff(x, y) #estimating coefficients
	print("Estimated coefficients:\nb0 = {} \
		\nb1 = {}".format(b[0], b[1])) 
 
	plot_regressionline(x, y, b) #plotting regression line

if __name__ == "__main__": 
	main() 
