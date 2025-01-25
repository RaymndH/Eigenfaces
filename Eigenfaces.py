import torch
import torchvision
from torchvision.io import read_image
import os, glob
import matplotlib.pyplot as plt
import time

def displayim(vec):
	if debug:
		plt.imshow(torch.reshape(vec,(h,w)), #interpolation='nearest',
		cmap='gray',vmin=0,vmax=255)
		plt.show()	

def face_projection(face,set_mean):
	print('very beginning')
	out=torch.empty(len(ul))
	tic=time.perf_counter()
	face[:]=face[:]-set_mean
	toc=time.perf_counter()
	print('time',toc-tic)
	print('inside face projection')
	for i in range(13):
		print(i)
		out[i]=torch.matmul(ul[:,i].T,face)
	return out
mps_device=torch.device("mps")
debug=True
print('starting')
M=40 #training set
Mp=13 #M prime, identification (largest eigenvalues)
print("M=",M)
w=320
h=243
A=torch.empty(h*w,M)
i=0

for infile in glob.glob("YalePNG/subject*"):
	if i==M:
		break
	temp=read_image(infile)
	new=torch.squeeze(temp)
	
	reshaped=torch.reshape(new,(h*w,1))
	#print(i)
	A[:,i]=reshaped.T
	i+=1

#Find mean
set_mean=torch.mean(A,1,True) #flatten along dim 1
#displayim(A[:,0])
tic=time.perf_counter()
A[:]=A[:]-set_mean
toc=time.perf_counter()
print('time',toc-tic)
#displayim(A[:,0])
#find svd, eigenvectors are columns of V
u,s,v=torch.svd(A)
#these eigenvectors determine linear combinations of training set faces
#print eigenvalues to display usefulness of eigenvectors
print('eigenvalues',s)
print('shape of u:',u.size())
#for future: calculate when steps are useful or not
#Eigenfaces
print(u[:,1])
#preprocessing?
ul=u[:,1:Mp]
print(ul[:,1])
#test with first face
print('projection:')
tic=time.perf_counter()
b=A[:,0]-set_mean
displayim(-b)
toc=time.perf_counter()
print('b',toc-tic)
newface=face_projection(A[:,0],set_mean)

displayim(newface)


