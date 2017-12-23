import numpy as np
import time
import random


# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.
###################################################################
# Example: Create a zeros vector of size 10 and store variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [0 for i in range(10)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros(10)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


z_1 = None
z_2 = None
################################################################
# 1. Create a zeros array of size (3,5) and store in variable z.
# Python
print("Answer 1 ")

python1StartTime = time.time()
z_1 = [[0 for x in range(5)] for y in range(3)] # using list comprehension to create 3 lists with 5 items each (3,5)
python1EndTime = time.time()

# NumPy
numPy1StartTime = time.time()

z_2 = np.zeros((3, 5))  # using the built in function of numpy to initiate a matrix of size 3x5
numPy1EndTime = time.time()

pyth_1=python1EndTime-python1StartTime
numpy_1=numPy1EndTime-numPy1StartTime

print('Answer 1 :- Python time: {0} sec.'.format(python1EndTime-python1StartTime))
print('Answer 1 :- NumPy time: {0} sec.'.format(numPy1EndTime-numPy1StartTime))

#################################################

# 2. Set all the elements in first row of z to 7.
# Python
print("Answer 2 ")

python2StartTime = time.time()

for i in range(5):
    z_1[0][i]=7
python2EndTime = time.time()

# NumPy
numPy2StartTime = time.time()
z_2[0, :] = 7
numPy2EndTime = time.time()

print('Answer 2 :- Python time: {0} sec.'.format(python2EndTime-python2StartTime))
print('Answer 2 :- NumPy time: {0} sec.'.format(numPy2EndTime-numPy2StartTime))

pyth_2=python2EndTime-python2StartTime
numpy_2=numPy2EndTime-numPy2StartTime
#####################################################
# 3. Set all the elements in second column of z to 9.
# Python
print("Answer 3 :")

python3StartTime = time.time()

for row in z_1:
    row[1]=9
python3EndTime = time.time()

# NumPy
numPy3StartTime = time.time()

z_2[:,1 ] = 9
numPy3EndTime = time.time()

print('Answer 3 :- Python time: {0} sec.'.format(python3EndTime-python3StartTime))
print('Answer 3 :- NumPy time: {0} sec.'.format(numPy3EndTime-numPy3StartTime))

pyth_3=python3EndTime-python3StartTime
numpy_3=numPy3EndTime-numPy3StartTime


#############################################################
# 4. Set the element at (second row, third column) of z to 5.
# Python
print("Answer 4 :")

python4StartTime = time.time()

z_1[1][3]=5
python4EndTime = time.time()


# NumPy
numPy4StartTime = time.time()

z_2[1][2]=5
numPy4EndTime = time.time()

print('Answer 4 :- Python time: {0} sec.'.format(python4EndTime-python4StartTime))
print('Answer 4 :- NumPy time: {0} sec.'.format(numPy4EndTime-numPy4StartTime))

pyth_4=python4EndTime-python4StartTime
numpy_4=numPy4EndTime-numPy4StartTime
##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
# Python
print("Answer 5 :")

python5StartTime = time.time()

x_1=[i for i in range(50 ,100)]
python5EndTime = time.time()

# NumPy
numPy5StartTime = time.time()

x_2=np.arange(50,100)
numPy5EndTime = time.time()
print('Answer 5 :- Python time: {0} sec.'.format(python5EndTime-python5StartTime))
print('Answer 5 :- NumPy time: {0} sec.'.format(numPy5EndTime-numPy5StartTime))

pyth_5=python5EndTime-python5StartTime
numpy_5=numPy5EndTime-numPy5StartTime

##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.

# Python
print("Answer 6 :")

python6StartTime = time.time()

y_1=[[0 for x in range(4)] for y in range(4)]
k=0
for i in range(4):
    for j in range(4):
        y_1[i][j]=k
        k=k+1
python6EndTime = time.time()

# NumPy
numPy6StartTime = time.time()


y_2=np.arange(0,16).reshape(4,4)
numPy6EndTime = time.time()

print('Answer 6 :- Python time: {0} sec.'.format(python6EndTime-python6StartTime))
print('Answer 6 :- NumPy time: {0} sec.'.format(numPy6EndTime-numPy6StartTime))

pyth_6=python6EndTime-python6StartTime
numpy_6=numPy6EndTime-numPy6StartTime
##############

print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.
# Python
print("Answer 7 :")

python7StartTime = time.time()


tmp_1 = [[0 for x in range(5)] for y in range(5)]

for i in range(5):
    for j in range(5):
        if i==0 or i==4 or j==0 or j==4:
         tmp_1[i][j]=1
        else:
            tmp_1[i][j]=0
python7EndTime = time.time()

# NumPy

numPy7StartTime = time.time()
tmp_2 = np.zeros((3,3))
tmp_2 = np.pad(tmp_2, pad_width=1, mode='constant', constant_values=1)

numPy7EndTime = time.time()

print('Answer 7:- Python time: {0} sec.'.format(python7EndTime-python7StartTime))
print('Answer 7 :- NumPy time: {0} sec.'.format(numPy7EndTime-numPy7StartTime))

pyth_7=python7EndTime-python7StartTime
numpy_7=numPy7EndTime-numPy7StartTime

##############
print(tmp_1)
print(tmp_2)
##############
a_1 = None; a_2 = None
b_1 = None; b_2 = None
c_1 = None; c_2 = None
#############################################################################################
# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
# Python
print("Answer 8 :")
python8StartTime = time.time()
a_1 =[[0 for x in range(100)] for y in range(50)]
k=0
for i in range(50):
    for j in range(100):
            a_1[i][j]=k
            k=k+1
python8EndTime = time.time()
# NumPy
numPy8StartTime = time.time()
a_2=np.arange(0,5000).reshape(50,100).astype(int)
numPy8EndTime = time.time()
print(a_1)
print(a_2)
print('Answer 8 :- Python time: {0} sec.'.format(python8EndTime-python8StartTime))
print('Answer 8 :- NumPy time: {0} sec.'.format(numPy8EndTime-numPy8StartTime))

pyth_8=python8EndTime-python8StartTime
numpy_8=numPy8EndTime-numPy8StartTime
###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
print("Answer 9 :")
python9StartTime = time.time()
b_1=[[0 for x in range(200)] for y in range(100)]
k=0
for i in range(100):
    for j in range(200):
            b_1[i][j]=k
            k=k+1
python9EndTime = time.time()
print(b_1)
# NumPy
numPy9StartTime = time.time()
b_2=np.arange(0,20000).reshape(100,200).astype(int)
print(b_2)
numPy9EndTime = time.time()
print('Answer 9 :- Python time: {0} sec.'.format(python9EndTime-python9StartTime))
print('Answer 9 :- NumPy time: {0} sec.'.format(numPy9EndTime-numPy9StartTime))

pyth_9=python9EndTime-python9StartTime
numpy_9=numPy9EndTime-numPy9StartTime

#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.
# Python
print("Answer 10 :")
python10StartTime = time.time()
c_1=[[0 for x in range(200)] for y in range(50)]
for row in range(len(a_1)):
    for column in range(len(b_1[0])):
        for k in range(len(b_1)):
            c_1[row][column]+=a_1[row][k]*b_1[k][column]
python10EndTime = time.time()
print(c_1)
# NumPy
numPy10StartTime = time.time()
c_2=np.dot(a_2,b_2)
numPy10EndTime = time.time()
print(c_2)
print('Answer 10 :- Python time: {0} sec.'.format(python10EndTime-python10StartTime))
print('Answer 10 :- NumPy time: {0} sec.'.format(numPy10EndTime-numPy10StartTime))

pyth_10=python10EndTime-python10StartTime
numpy_10=numPy10EndTime-numPy10StartTime
d_1 = None; d_2 = None
################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
# Python
print("Answer 11 :")
python11StartTime = time.time()
d_1=[[random.random() for x in range(3)] for y in range(3)]
for row in range(3):
    for col in range(3):
        d_1[row][col]=(d_1[row][col]-min(d_1[row]))/(max(d_1[row])-min(d_1[row]))
python11EndTime = time.time()

# NumPy
np.seterr(divide='ignore', invalid='ignore')
numPy11StartTime = time.time()
d_2=np.random.random((3,3))
max_d2=d_2.max()
min_d2=d_2.min()
d_2=(d_2-max_d2)/(d_2-min_d2)
numPy11EndTime = time.time()

print('Answer 11 :- Python time: {0} sec.'.format(python11EndTime-python11StartTime))
print('Answer 11 :- NumPy time: {0} sec.'.format(numPy11EndTime-numPy11StartTime))

pyth_11=python11EndTime-python11StartTime
numpy_11=numPy11EndTime-numPy11StartTime
##########
print(d_1)
print(d_2)
#########


################################################
# 12. Subtract the mean of each row of matrix a.
# Python
print("Answer 12")

python12StartTime = time.time()

new_a_1 =a_1

for row in range(50):
    for col in range(100):
        new_a_1[row][col]=a_1[row][col] -(sum(a_1[row]) / len(a_1[row]))

print(new_a_1)
python12EndTime = time.time()

# NumPy

numPy12StartTime = time.time()
a2mean=a_2.mean(axis=1,keepdims=True)
new_a_2=a_2-a2mean
numPy12EndTime = time.time()

print(new_a_2)
print('Answer 12 :- Python time: {0} sec.'.format(python12EndTime-python12StartTime))
print('Answer 12 :- NumPy time: {0} sec.'.format(numPy12EndTime-numPy12StartTime))

pyth_12=python12EndTime-python12StartTime
numpy_12=numPy12EndTime-numPy12StartTime

###################################################
# 13. Subtract the mean of each column of matrix b.
# Python
print("Answer 13")
new_b_1=b_1
python13StartTime = time.time()
b_1_mean=[sum([b_1[j][i] for j in range(len(b_1))])/len(b_1) for i in range(len(b_1[0]))]
new_b_1=[[b_1[i][j]-b_1_mean[j] for j in range(len(b_1[0]))] for i in range(len(b_1))]
print(new_b_1)
python13EndTime = time.time()
# NumPy
numPy13StartTime = time.time()
new_b_2 = b_2-b_2.mean(axis=0)
numPy13EndTime = time.time()
print(new_b_2)

################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

print('Answer 13 :- Python time: {0} sec.'.format(python13EndTime-python13StartTime))
print('Answer 13 :- NumPy time: {0} sec.'.format(numPy13EndTime-numPy13StartTime))

pyth_13=python13EndTime-python13StartTime
numpy_13=numPy13EndTime-numPy13StartTime
e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
# Python
print("Answer 14")
python14StartTime = time.time()

e_1 = [[c_1[j][i]+5 for j in range(len(c_1))] for i in range(len(c_1[0]))]
print(e_1)
python14EndTime = time.time()

# NumPy
numPy14StartTime = time.time()

e_2b=c_2.transpose()
e_2a=np.zeros_like(e_2b)
e_2a.fill(5)
e_2=np.add(e_2a,e_2b)
numPy14EndTime = time.time()
print(e_2)
##################
print(np.sum(e_1 == e_2))
##################
print('Answer 14 :- Python time: {0} sec.'.format(python14EndTime-python14StartTime))
print('Answer 14 :- NumPy time: {0} sec.'.format(numPy14EndTime-numPy14StartTime))

pyth_14=python14EndTime-python14StartTime
numpy_14=numPy14EndTime-numPy14StartTime

#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
# Python
python15StartTime = time.time()

f_1 = []
for i in range(len(e_1)):
    for j in range(len(e_1[0])):
        f_1.append(e_1[i][j])
print("Shape of f_1: ")
print(len(f_1))

python15EndTime = time.time()

# NumPy
numPy15StartTime = time.time()
f_2=e_2.flatten()
print("Shape of f_2:")
print(f_2.shape)
numPy15EndTime = time.time()
print('Answer 15 :- Python time: {0} sec.'.format(python15EndTime-python15StartTime))
print('Answer 15 :- NumPy time: {0} sec.'.format(numPy15EndTime-numPy15StartTime))

pyth_15=python15EndTime-python15StartTime
numpy_15=numPy15EndTime-numPy15StartTime


