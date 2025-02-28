#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:53:03 2025

@author: axel
"""
import numpy as np
import uncertainties as u
import numba
import iminuit 
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import uncertainties.umath as um

adc_bins = 10/(2**12)
adc_err_u = adc_bins/3**0.5

R_REF_4700 = u.ufloat(4670, 10/3**0.5)

def read_data(filename, name = "", plot_raw = False, plot_errorbar = False, saveraw = "raw.png", saveerrorbar ="err.pdf"):
    '''NICHT FÜR DATEN AUS DEM SPEICHEROSZILLOSKOP \n
    ließt daten ein, mittelt sie und gibt die binmittelwerte und die fehler auf die mittelwerte aus \n
    plottet rohdaten und mittelwerte mit fehlern'''
    
    if name == "":
        name = filename
        
    #with open(filename), "r") as file:
     #   lines = file.readlines()
      #  lines
        
        
    Data = np.genfromtxt(filename)
    #print(Data.shape)
    
    Data_mean = np.zeros((Data.shape[0]//11,Data.shape[1] ))
    Data_err = np.zeros((Data.shape[0]//11,Data.shape[1] ))
    #print(Data_err.shape)
    if plot_raw == True:
        plt.scatter(Data[:,0], Data[:,1])
        plt.title(name)
        #plt.ylabel(
        plt.savefig(saveraw)
        plt.show()

    for i in range(Data.shape[0]//11):
        Data_mean[i,0] = np.mean(Data[i*11:(i+1)*11,0])
        Data_mean[i,1] = np.mean(Data[i*11:(i+1)*11,1])
    #print(Data[i*11:(i+1)*11,1])
    

        Data_err[i,0] = np.std(Data[i*11:(i+1)*11,0], ddof=1)/11**0.5
        Data_err[i,1] = np.std(Data[i*11:(i+1)*11,1], ddof=1)/11*0.5
    #print(Data_err<adc_err_u)
    Data_err = Data_err*[Data_err>adc_err_u]+ adc_err_u*np.array(Data_err<adc_err_u)
    #Data_mean[0,i] = np.mean(Data[0,i*11:(i+1)*11])
    
    Data_err = Data_err[0] 
    if plot_errorbar == True:
        plt.errorbar(Data_mean[:,0], Data_mean[:,1], Data_err[:,1], Data_err[:,0], fmt=".")
        plt.title(name + " Means & Error")
        plt.savefig(saveerrorbar)
        plt.show()

    #plt.plot(Data_mean[:,0], Data_mean[:,1])
    #plt.show()
    
    return Data_mean, Data_err


def lin(x, m,b):
    "lineare fitfunktion"
    return m*x+b

def chi2(fit_func, x, y, xerr, yerr, a, c):
    'chi2 für linearen fit'
    chi2_value = 0
    for i in range(len(x)):
        model = fit_func(x[i], a, c)
        chi2_value += ((y[i] - model) / np.sqrt(yerr[i]**2 + (a * xerr[i])**2))**2
    return chi2_value



#%% linfit
def linfit_res(res_data, res_means):
    'linear fitting function for resistors'
    
    r = unp.uarray(res_data,res_means)
    i = r[:,1]/R_REF_4700

    
    chi2_r = lambda m, b: chi2(lin ,y = res_data[:,0], x = unp.nominal_values(i[:]), yerr=res_means[:,0], xerr = unp.std_devs(i[:]), a = m , c=b)

    m0 = iminuit.Minuit(chi2_r, m = 1, b = 0)

    m0.migrad()
    m0.hesse()
    return m0.values, m0.errors, m0.fval, m0.fval/2
    

#%% linfit für R

r1d, r1e = read_data("r_470.txt", plot_raw= True, plot_errorbar=True)
print(linfit_res(r1d, r1e))

r2d, r2e = read_data("r_1000.txt", plot_raw= True, plot_errorbar=True)
print(linfit_res(r2d, r2e))

r3d, r3e = read_data("r_10000.txt", plot_raw= True, plot_errorbar=True)
print(linfit_res(r3d, r3e))

#%%

# read_data("eingang_b107_vce0.txt", plot_raw= True, plot_errorbar=True)
# IB, eIB = read_data("IB_IC_b107_NEU.txt", plot_raw= True, plot_errorbar=True)
# read_data("UBE_IC_b107_NEU.txt", plot_raw= True, plot_errorbar=True)
# u600mv, u6e = read_data("ausgang_ube600mv.txt", plot_raw= True, plot_errorbar=True)
# u700mv,u7e = read_data("ausgang_ube700mv.txt", plot_raw= True, plot_errorbar=True)
# u100mv, u1e = read_data("ausgang_ube100mv.txt", plot_raw= True, plot_errorbar=True)
# u800mv,u8e = read_data("ausgang_ube800mv.txt", plot_raw= True, plot_errorbar=True)
#Data = np.genfromtxt("c_entladekurve.txt")

#plt.scatter(Data[:,0],Data[:,1], )
#plt.scatter(Data[:,0],Data[:,2], )

#plt.show()

z_diode,uz_d_err = read_data("z_diode_100.txt", plot_raw= True, plot_errorbar=True)
z_diode,uz_d_err = read_data("z_diode_1000.txt", plot_raw= True, plot_errorbar=True)

gleichrichter = np.genfromtxt("Gleichrichter_1000R_Neu.txt")
plt.scatter(gleichrichter[:,0], gleichrichter[:,1], marker = ".")
plt.scatter(gleichrichter[:,0], gleichrichter[:,2], marker = ".")
plt.show()

#%%transistorkennlinien
IB_IC, IB_ICerr = read_data("IB_IC_B107_NEU_NEU.txt", plot_raw= True, plot_errorbar=False)
plt.scatter(IB_IC[0:5000,0],IB_IC[0:5000,1])
#plt.show()

for i in range(8):
    print((IB_IC[i*150+4000,0])/10000)
    print(((IB_IC[i*150+4000,1])/470)/((IB_IC[i*150+4000,0])/10000))
    plt.axhline(IB_IC[i*150+4000,1])
    
plt.plot(IB_IC[0:5000,0], 200*470/10000*IB_IC[0:5000,0], color = "r")
plt.plot(IB_IC[0:5000,0], 400*470/10000*IB_IC[0:5000,0], color = "r")
plt.show()
# plt.plot(r1d[:3,0],r1d[:3,1])
# plt.plot(r2d[:3,0],r2d[:3,1])
# plt.plot(r3d[:3,0],r3d[:3,1])
# plt.errorbar(0,0, adc_err_u, adc_err_u)
# plt.show()

# plt.scatter(IB[:100,0],IB[:100,1])
# plt.show()

read_data()

# plt.scatter(u100mv[:,0],u100mv[:,1])
# plt.scatter(u600mv[:,0],u600mv[:,1])
# plt.scatter(u700mv[:,0],u700mv[:,1])
# plt.scatter(u800mv[:,0],u800mv[:,1])
# plt.plot()