import numpy as np 
import math
import operator
import matplotlib.pyplot as plt

# --------------------------------------- 
#   	       ANN Model
# ---------------------------------------
def importWB(fModel, nlayer):
	wlst1 = []
	blst1 = []
	wlst2 = []
	blst2 = []
	iwlst = []	
	for i in range(0, nlayer): # <-<-<-<- # of layer
		warray1 = np.genfromtxt(fModel+'w1_'+str(i)+'.csv', delimiter=',').tolist()
		barray1 = np.genfromtxt(fModel+'b1_'+str(i)+'.csv', delimiter=',').tolist()
		warray2 = np.genfromtxt(fModel+'w2_'+str(i)+'.csv', delimiter=',').tolist()
		barray2 = np.genfromtxt(fModel+'b2_'+str(i)+'.csv', delimiter=',').tolist()
		iwarray = np.genfromtxt(fModel+'iw_'+str(i)+'.csv', delimiter=',').tolist()
		if i == 0 and isinstance(warray1[0], float):
			warray1 = [[e] for e in warray1]
		if i > 0 and isinstance(warray1[0], float):
			warray1 = [warray1]
		if isinstance(barray1, float):
			barray1 = [barray1]
		if i == 0 and isinstance(warray2[0], float):
			warray2 = [[e] for e in warray2]
		if i > 0 and isinstance(warray2[0], float):
			warray2 = [warray2]
		if isinstance(barray2, float):
			barray2 = [barray2]
		if isinstance(iwarray, float):
			iwarray = [[iwarray]]
		wlst1.append(warray1)
		blst1.append(barray1)
		wlst2.append(warray2)
		blst2.append(barray2)		
		iwlst.append(iwarray)
	return (wlst1, blst1, wlst2, blst2, iwlst)


def sigmoid(x) :
	try:
		y = math.exp(-x)
	except OverflowError:
		y = float('inf')
	return 1.0 / (1.0 + y)

def tanh(x):
	return math.tanh(x)

def linear(x):
	return x

def activation(ai, x, yi):
	return (sum(map(operator.mul, ai, x)) + yi)

def activation2(ai, x):
	return (sum(map(operator.mul, ai, x)))

def gemv(weight, x, bias):
	z = []
	for i in range(0,len(bias)):
		z.append(activation(weight[i], x, bias[i]))
	return z

def gemv1(weight, x):
	z = []
	for i in range(0,len(weight)):
		z.append(activation2(weight[i], x))
	return z

def gemv2(iweight, y, weight, x, bias):
	z = []
	for i in range(0,len(bias)):
		z.append(activation2(iweight[i], y) + activation(weight[i], x, bias[i]))
	return z

def forwardprop(w1, b1, w2, b2, iw, x1, x2):
	if not b1:
		return map(operator.mul, x1, x2)
	else:
		x1 = map(tanh, gemv(w1[0], x1, b1[0]))
		x2 = map(sigmoid, gemv2(iw[0], x1, w2[0], x2, b2[0]))
		return forwardprop(w1[1:], b1[1:], w2[1:], b2[1:], iw[1:], x1, x2)

def make_dmm(fModel, nlyr):
	wb = importWB(fModel, nlyr)
	def current(Vg, Vd):
		v1 = [Vd * 2.5]
		v2 = [(Vg-0.2)*5]
		i = (forwardprop(wb[0], wb[1], wb[2], wb[3], wb[4], v1, v2)[0]) * 375
		i = i / (10**(-12 * (v2[0] + 0.75)) + 1)  
		return i
	return current

# --------------------------------------- 
#   	     plot and save
# ---------------------------------------

def plot(current, x_list, c_list, TYPE = "t", LOG = False, color = 'navy'):
	savelist = []
	for vc in c_list:
		l = []
		lNeg = [] # for log plot 
		for vx in x_list:
			if TYPE == "t":
				i = current(vx, vc) # Vg, Vd
			if TYPE == "f":
				i = current(vc, vx) # Vg, Vd
			if LOG:
				if i >= 0:
					l.append(i)
					lNeg.append(None)
				if i < 0:
					lNeg.append(-i)
					l.append(None)
			else:
				l.append(i)
		if LOG:
			plt.semilogy(x_list, l,  color = color, linewidth=2.0)
			plt.semilogy(x_list, lNeg,  color = color, linewidth=2.0, linestyle='--')
		else:
			plt.plot(x_list, l, color = color, linewidth=2.0)
	savelist.append(l)
	# plt.show()
	return savelist

def saveData(label,fData, c_list, r_list, save_list, scale, c_list_label):
	with open(label + "_" + fData + '_trained.csv', 'wb') as csvfile:
		iwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
		firstRow = [c_list_label]
		firstRow.extend(c_list)
		iwriter.writerow(firstRow)
		for i in range(len(r_list)): 
			row = []
			row.append(r_list[i])
			for j in range(len(c_list)):
				row.append(save_list[j][i] * scale)
			iwriter.writerow(row)


if __name__ == "__main__":
	pass