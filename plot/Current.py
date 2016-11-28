import numpy as np
import matplotlib.pyplot as plt
import csv
import math


class Current(object):

	def __init__(self, file_name, output_row):
		""" file_name is a string """
		assert isinstance(file_name, str), 'file name has to be a string'
		self.data = np.genfromtxt(file_name, dtype = float, 
			delimiter=',',skip_header = 1)
		# print(self.data[:,0])
		# find a way to parse n-DG-pin-TFET-no..._Cgd.csv
		# in order to know whether it is Cgs, Cgg, or Cgd
		# also save the part before Cgd as indicator
		self.fileName = file_name;
		# [VbgList, VtgList, VdsList]
		self.volList = self._createVolList()
		# print(self.volList[2])
		# Dictionary: Vbg, Vtg, Vds -> total current
		self.currentDict = self._createCurrentDict(output_row)
		# print(self.currentDict[(0, 10, -200)])


	def plotFamily(self, DG = False, LOG = False, c_list = 0, color = 'navy', shape = 'o'):
		Vbg = 0
		vdslist = []
		for v in self.volList[2]:
			vdslist.append(v/1000.0)
		if c_list == 0:
			c_list = [v/1000 for v in self.volList[1]]
		for vtg in c_list:
			Vtg = int(vtg * 1000)
			if LOG:
				# plt.semilogy(vdslist, self._plotHelper(Vbg, Vtg, self.volList[2], DG, LOG), 'ro')
				pass
			else:	
				# plt.plot(vdslist, self._plotHelper(Vbg, Vtg, self.volList[2], DG, LOG), 'ro')
				plt.scatter(vdslist, self._plotHelper(Vbg, Vtg, self.volList[2], DG, LOG), 
					s = 40, marker = shape, linewidth='2', facecolors = 'none', edgecolors = color)

		# plt.show()

	def plotTransfer(self, DG = False, LOG = False, c_list = 0, color = 'navy', shape = 'o'):
		Vbg = 0
		vtglist = []
		for v in self.volList[1]:
			vtglist.append(v/1000.0)
		if c_list == 0:
			c_list = [v/1000 for v in self.volList[2]]
		for vds in c_list:
			Vds = int(vds * 1000)
			if LOG:
				plt.semilogy(vtglist, self._plotHelper(Vbg, self.volList[1], Vds, DG, LOG), "r-")
			else:
				# plt.plot(vtglist, self._plotHelper(Vbg, self.volList[1], Vds, DG, LOG), "r-")
				plt.scatter(vtglist, self._plotHelper(Vbg, self.volList[1], Vds, DG, LOG), 
					s = 40, marker = shape, linewidth='2', facecolors = 'none', edgecolors = color)

		# plt.show()

	def _plotHelper(self, vbg, vtg, vds, DG = False, LOG = False):
		""" pre: vbg is a single value;
			if vtg/vds is a list, create the according capa list,
			if Double Gate (DG is True, Vbg is connected with Vtg
		"""
		capa = []
		if isinstance(vtg, list) is True and isinstance(vds, list) is False:
			for vtgSingle in vtg:
				if DG is True:
					vbg = vtgSingle # overide Vbg
				tempCapa = self.currentDict[(vbg, vtgSingle, vds)]
				if LOG:
					if tempCapa > 0:
						capa.append(tempCapa*15e-5)
					else:
						capa.append(None)
				else:
					capa.append(tempCapa*15e-5)

		if isinstance(vtg, list) is False and isinstance(vds, list) is True:
			if DG is True:
				vbg = vtg
			for vdsSingle in vds:
				tempCapa = self.currentDict[(vbg, vtg, vdsSingle)]
				if LOG:
					if tempCapa > 0:
						capa.append(tempCapa*15e-5)
					else:
						capa.append(None)
				else:
					capa.append(tempCapa*15e-5)
		# print capa
		return capa

	def saveTransfer(self, fileName, vdsList, scale, DG = False):
		""" vdsList contains all the vds value
		we like to show.
		Data are saved in the form of:

		    Vds Vds
		Vtg Cap Cap
			. 	.   .   .
			.   .   .   .
			.   .   .   .

		easy for pasting into a plotting software.
		"""
		Vbg = 0
		capaMat = []
		for Vds in vdsList:
			capaMat.append( self._plotHelper(Vbg, self.volList[1], int(1000*Vds), DG) )

		# Start to write to csv
		with open('transfer_' + fileName + '.csv', 'wb') as csvfile:
			iwriter = csv.writer(csvfile, dialect='excel')
			# first row
			firstRow = ['Vds(mV)']
			firstRow.extend(vdsList)
			iwriter.writerow(firstRow)

			for i in range(len(self.volList[1])): # Vtg
				row = []
				row.append(self.volList[1][i]/1000.0)
				for j in range(len(vdsList)): # Vds
					row.append(capaMat[j][i] * scale)
				iwriter.writerow(row)


	def saveFamily(self, fileName, vtgList, scale, DG = False):
		Vbg = 0
		capaMat = []
		for Vtg in vtgList:
			capaMat.append( self._plotHelper(Vbg, int(1000*Vtg), self.volList[2], DG) )

		# Start to write to csv
		with open('family_' + fileName + '.csv', 'wb') as csvfile:
			iwriter = csv.writer(csvfile, dialect='excel')
			# first row
			firstRow = ['Vtg(mV)']
			firstRow.extend(vtgList)
			iwriter.writerow(firstRow)

			for i in range(len(self.volList[2])): # Vds
				row = []
				row.append(self.volList[2][i]/1000.0)
				for j in range(len(vtgList)): # Vds
					row.append(capaMat[j][i] * scale)
				iwriter.writerow(row)

	def trainingData(self, fileName, type):
		pass

	def _createCurrentDict(self, index):
		''' (Vbg, Vtg, Vds) -> current '''
		currentDict = {}	

		for i in range(self.data.shape[0]):		
			volTuple = (int(round(self.data[i][1],3)*1000), 
				int(round(self.data[i][0],3)*1000), 
				int(round(self.data[i][2],3)*1000) )
			currentDict[volTuple] = self.data[i][index]

		return currentDict


	def _createVolList(self):
		""" create the voltage list 
			[VbgList, VtgList, VdsList]
			Unit: 1 mV """
		volList = []

		for i in [1,0,2]: # per a voltage (e.g. Vtg)
			s = set(self.data[:, i]) # the set of all voltage value of Vtg
			tempVolList = []
			for vol in s:	# for each voltage value of Vtg
				tempVolList.append(int(round(vol,3)*1000))

			tempVolList.sort()
			volList.append( tempVolList )
 
		return volList 


	def _saveHelper(self, majorList, minorList):
		pass


if __name__ == "__main__":
	pass

