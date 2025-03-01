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

adc_bins = 20/(2**12)
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
#%% linfit

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




def linfit_res(res_data, res_std):
    'linear fitting function for resistors'
    
    r = unp.uarray(res_data,res_std)
    i = r[:,1]/R_REF_4700

    
    chi2_r = lambda m, b: chi2(lin ,y = res_data[:,0], x = unp.nominal_values(i[:]), yerr=res_std[:,0], xerr = unp.std_devs(i[:]), a = m , c=b)

    m0 = iminuit.Minuit(chi2_r, m = 1, b = 0)

    m0.migrad()
    m0.hesse()
    return m0.values, m0.errors, m0.fval, len(res_data) - 2, m0.fval/(len(res_data) - 2)
    
def linfit_plot(res_data, res_std, name = "", filename = "test.pdf"):
    val, err, chisq, dof, chindof = linfit_res(res_data, res_std)
    res = u.ufloat(val["m"],err["m"])
    b = u.ufloat(val["b"],err["b"])
    
    
    
    fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,7), layout = "tight",sharex=True, gridspec_kw={'height_ratios': [5, 2]})
    
    r = unp.uarray(res_data,res_std)
    i = r[:,1]/R_REF_4700
    #print(i)
    
    fity = lin(unp.nominal_values(i), val["m"],val["b"])
    fityplus = lin(unp.nominal_values(i), val["m"]+err["m"],val["b"]+err["b"])
    fityminus = lin(unp.nominal_values(i), val["m"]-err["m"],val["b"]-err["b"])
    
    ax[0].plot(unp.nominal_values(i), fity, label = "Fit", color = "r", zorder = 4)
    ax[0].errorbar(unp.nominal_values(i),res_data[:,0], res_std[:,0], unp.std_devs(i), fmt =".", elinewidth=0.4, label = "Messwerte", zorder =2)
    ax[0].fill_between(unp.nominal_values(i), fityminus, fityplus, alpha=.5, linewidth=0, label = "err_fit", color = "r", zorder = 3)
    #ax[0].plot(unp.nominal_values(i), fity, label = "Fit")
    
    ax[0].title.set_text("Lineare Regression & Residuenplot für R = " + name)
    #ax[0].set_xlabel('$I$ [$A$] ')
    ax[0].set_ylabel('$U_R$ [$I$] ')
    ax[0].legend(fontsize = 13)
    #ax[0].axhline(y=0., color='black', linestyle='--')
    
    sigmaRes = np.sqrt((val["m"]*unp.std_devs(i))**2 + res_std[:,0]**2)
    
    
    ax[1].axhline(y=0., color='black', linestyle='--', zorder = 4)
    ax[1].fill_between(unp.nominal_values(i), fity-fityminus, fity-fityplus, linewidth = 0, alpha = .5, label = "err_fit", color = "r", zorder = 3)
    ax[1].errorbar(unp.nominal_values(i), res_data[:,0]-fity, sigmaRes, zorder = 2, fmt =".", elinewidth=0.4, label = "Residuen" )
    #ax[1].fill_between(unp.nominal_values(i), fity-fityminus, fity-fityplus, alpha=0, linewidth = 0, label = "err_fit", color = "r")
    #ax[1].axhline(y=0., color='black', linestyle='--')
    ax[1].set_ylabel('$U_R- R_fit*I$ [$I$] ')
    ax[1].set_xlabel('$I$ [$A$] ')
    ymax = max([abs(x) for x in ax[1].get_ylim()])
    ax[1].set_ylim(-ymax, ymax)
    ax[1].legend(fontsize = 13)
    fig.text(0.5,0, f'R = ({res})$\Omega$ , b = ({b})V, chi2/dof = {chisq:.1f} / {dof} = {chindof:.3f} ', horizontalalignment = "center")
    fig.subplots_adjust(hspace=0.0)
    
    
    plt.savefig(filename)
    plt.show()
    return True
    

    
#%% linfit für R

r1d, r1e = read_data("r_470.txt", plot_raw= True, plot_errorbar=True)
linfit_plot(r1d, r1e, "$470 \Omega$", "R470_fit.pdf")

r2d, r2e = read_data("r_1000.txt", plot_raw= True, plot_errorbar=True)
linfit_plot(r2d, r2e, "$1000 \Omega$", "R1000_fit.pdf")

r3d, r3e = read_data("r_10000.txt", plot_raw= True, plot_errorbar=True)
linfit_plot(r3d, r3e, "$10000 \Omega$", "R10000_fit.pdf")

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

#%%zdiode

def linfit_z(res_data, res_std):
    'linear fitting function for Q with Zdiode'
    
    r = unp.uarray(res_data,res_std)
    #i = r[:,1]/R_REF_4700

    
    chi2_r = lambda m, b: chi2(lin ,y = res_data[:,0], x = unp.nominal_values(r[:]), yerr=res_std[:,0], xerr = unp.std_devs(r[:]), a = m , c=b)

    m0 = iminuit.Minuit(chi2_r, m = 1, b = 0)

    m0.migrad()
    m0.hesse()
    return m0.values, m0.errors, m0.fval, len(res_data) - 2, m0.fval/(len(res_data) - 2)

def z_diode_plot(zdata, zerr):
    fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,7), layout = "tight")

    ax[0].errorbar(zdata[:,0], zdata[:,1], zerr[:,1], zerr[:,0], fmt =".")
    
    arbeitspunkt = -3
    zdata_sliced = zdata[np.abs(zdata[:,0]-arbeitspunkt)<0.05]
    zerr_sliced = zerr[np.abs(zdata[:,0]-arbeitspunkt)<0.05]
    
    
    
    ax[1].errorbar(zdata_sliced[:,0], zdata_sliced[:,1], zerr_sliced[:,1], zerr_sliced[:,0], fmt =".")
    plt.show()
    
    

    return True

z_diode_100,uz_d_err_100 = read_data("z_diode_100.txt", plot_raw= True, plot_errorbar=True)
z_diode_100[:,0] = -z_diode_100[:,0]
z_diode_1000,uz_d_err_1000 = read_data("z_diode_1000.txt", plot_raw= True, plot_errorbar=True)
z_diode_1000[:,0] = -z_diode_1000[:,0]
#plt.plot(z_diode_1000[:,0],z_diode_1000[:,1])
#plt.plot(z_diode_100[:,0],z_diode_100[:,1])

z_diode_plot(z_diode_100, uz_d_err_100)
z_diode_plot(z_diode_1000, uz_d_err_1000)


#%%gleichrichter
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