## ANN compact model by Mingda Li


from dmm import *
from Current import *
from pylab import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
rcParams['figure.figsize'] = 6.5, 10
rcParams['axes.linewidth'] = 2 # set the value globally
fig, ax = plt.subplots()
xminorLocator = AutoMinorLocator()
yminorLocator = AutoMinorLocator()

# ANN model
dmm = make_dmm('../model/', 2) # num. of layers
# original data
fTrain = 'original_data/pTrain'
fTest = 'original_data/pTest'
c = Current(fTrain +".csv", 4)
t = Current(fTest +".csv", 4)
# plot
x_list = [round(i*0.0005 - 0.1, 3) for i in range(0, 1000)]
c_list = [round(i*0.03 + 0.01, 3) for i in range(0, 14)] + [0]# Train
t_list = [round(i*0.03 - 0.005, 3) for i in range(1, 14)] + [0.005]# Test

# Nice plot
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tick_params(axis='both', which='minor', labelsize=0)

# ------------------- Fig 6.c --------------------------- 
c.plotFamily(c_list = c_list, color = '#1496BB', shape = 'o')
t.plotFamily(c_list = t_list, color = '#C02F1D', shape = '^')
plot(dmm, x_list, t_list + c_list, TYPE = 'f', LOG = False, color = '#093145')
plt.ylim([-50,350])
plt.xlim([-0.1,0.4])
# y ticker
ax.yaxis.set_minor_locator(yminorLocator) # need to be commented out for log plot
# ------------------------------------------------------- 

# ------------------- Fig 6.d --------------------------- 
# c.plotTransfer(c_list = c_list, color = '#1496BB', shape = 'o')
# t.plotTransfer(c_list = t_list, color = '#C02F1D', shape = '^')
# plot(dmm, x_list, t_list + c_list, TYPE = 't', LOG = False, color = '#093145')
# plt.ylim([-50,350])
# plt.xlim([-0.1,0.4])
# # y ticker
# ax.yaxis.set_minor_locator(yminorLocator) # need to be commented out for log plot
# ------------------------------------------------------- 

# ------------------- Fig 6.e --------------------------- 
# c.plotFamily(c_list = c_list, color = '#1496BB', shape = 'o')
# t.plotFamily(c_list = t_list, color = '#C02F1D', shape = '^')
# plot(dmm, x_list, t_list + c_list, TYPE = 'f', LOG = False, color = '#093145')
# plt.ylim([-50,150])
# plt.xlim([-0.05,0.05])
# # y ticker
# ax.yaxis.set_minor_locator(yminorLocator) # need to be commented out for log plot
# ------------------------------------------------------- 

# ------------------- Fig 6.e insert -------------------- 
# plot(dmm, x_list, t_list + c_list, TYPE = 'f', LOG = False, color = '#093145')
# plt.ylim([-1,1])
# plt.xlim([-0.0045,0.0045])
# # y ticker
# ax.yaxis.set_minor_locator(yminorLocator) # need to be commented out for log plot
# ------------------------------------------------------- 

# ------------------- Fig 6.d --------------------------- 
# c.plotTransfer(c_list = c_list, color = '#1496BB', shape = 'o')
# t.plotTransfer(c_list = t_list, color = '#C02F1D', shape = '^')
# plot(dmm, x_list, t_list + c_list, TYPE = 't', LOG = True, color = '#093145')
# plt.ylim([1e-4,1e3])
# plt.xlim([-0.1,0.4])
# ------------------------------------------------------- 

# x ticker
ax.xaxis.set_minor_locator(xminorLocator)
plt.tick_params(which='both', width=2)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=4)
plt.show()
# plt.savefig('1.eps', format='eps', dpi=1000)

# Family
# c.saveFamily(fData, vtg_list, 15e-5)
# saveData("family", "tan", c_list, x_list, tan_save, 1, "Vtg")

# transfer
# c.saveTransfer(fData, vds_list, 15e-5)
# saveData("transfer", "tan", c_list, x_list, tan_save, 1, "Vds")