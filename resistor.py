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
from iminuit import cost, Minuit
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import uncertainties.umath as um

adc_bins = (10.622260497*2)/(2**16)
adc_err_u = adc_bins/3**0.5 * 1/2 # TODO RICHTIGEN FEHLER FINDEN

R_REF_4700 = u.ufloat(4670, 0)
R_REF_4700_err = 10/12**0.5

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
    
    

        Data_err[i,0] = np.std(Data[i*11:(i+1)*11,0], ddof=1)
        Data_err[i,1] = np.std(Data[i*11:(i+1)*11,1], ddof=1)
    #print(Data_err<adc_err_u)
    Data_err = Data_err*[Data_err>adc_err_u]+ adc_err_u*np.array(Data_err<adc_err_u)
    #Data_mean[0,i] = np.mean(Data[0,i*11:(i+1)*11])
    Data_err = Data_err/(11)**0.5
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
    
    i = r[:,1]/(R_REF_4700+R_REF_4700_err)
    
    chi2_r = lambda m, b: chi2(lin ,y = res_data[:,0], x = unp.nominal_values(i[:]), yerr=res_std[:,0], xerr = unp.std_devs(i[:]), a = m , c=b)

    m0plus = iminuit.Minuit(chi2_r, m = 1, b = 0)

    m0plus.migrad()
    m0plus.hesse()
    
    i = r[:,1]/(R_REF_4700-R_REF_4700_err)
    
    chi2_r = lambda m, b: chi2(lin ,y = res_data[:,0], x = unp.nominal_values(i[:]), yerr=res_std[:,0], xerr = unp.std_devs(i[:]), a = m , c=b)

    m0minus = iminuit.Minuit(chi2_r, m = 1, b = 0)

    m0minus.migrad()
    m0minus.hesse()
    return m0.values, m0.errors, m0.fval, len(res_data) - 2, m0.fval/(len(res_data) - 2), m0plus.values, m0minus.values
    
def linfit_plot(res_data, res_std, name = "", filename = "test.pdf"):
    val, err, chisq, dof, chindof, valplus, valminus = linfit_res(res_data, res_std)
    res = u.ufloat(val["m"],err["m"])
    b = u.ufloat(val["b"],err["b"])
    
    
    
    fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,7), layout = "tight",sharex=True, gridspec_kw={'height_ratios': [5, 2]})
    
    r = unp.uarray(res_data,res_std)
    i = r[:,1]/R_REF_4700
    #print(i)
    
    fity = lin(unp.nominal_values(i), val["m"],val["b"])
    fityplus = lin(unp.nominal_values(i), val["m"]+err["m"],val["b"]+err["b"])
    fityminus = lin(unp.nominal_values(i), val["m"]-err["m"],val["b"]-err["b"])
    fitsysplus = lin(unp.nominal_values(i), valplus["m"],valplus["b"])
    fitsysminus = lin(unp.nominal_values(i), valminus["m"],valminus["b"])
    
    ax[0].plot(unp.nominal_values(i), fity, label = "Fit", color = "r", zorder = 4)
    ax[0].plot(unp.nominal_values(i), fitsysminus, label = "Fit - Lower $R_{Ref}$ ", color = "grey",linestyle = "--", zorder = 3)
    ax[0].plot(unp.nominal_values(i), fitsysplus, label = "Fit - Higher $R_{Ref}$ ", color = "grey",linestyle = "--", zorder = 3)
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
    sigma_R_sys = val["m"] - valminus["m"]
    fig.text(0.5,0, f'R = ({res})$\Omega$,'+' $\sigma_{R,sys}$ = '+f'{sigma_R_sys:0.2f}$\Omega$ , b = ({b})V, chi2/dof = {chisq:.1f} / {dof} = {chindof:.3f} ', horizontalalignment = "center")
    fig.subplots_adjust(hspace=0.0)
    
    #print(valplus, valminus)
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

def linfitr(res_data, res_std):
    'linear fitting function for resistors'
    
    r = unp.uarray(res_data,res_std)
    i = r[:,1]
    
    chi2_r = lambda m, b: chi2(lin ,x = res_data[:,0], y = unp.nominal_values(i[:]), xerr=res_std[:,0], yerr = unp.std_devs(i[:]), a = m , c=b)

    m0 = iminuit.Minuit(chi2_r, m = 1, b = 0)

    m0.migrad()
    m0.hesse()
    return m0.values, m0.errors, m0.fval, len(res_data) - 2, m0.fval/(len(res_data) - 2)


plt.errorbar(r1d[:5,0],r1d[:5,1],r1e[:5,1],r1e[:5,0], fmt = ".", label = "$R = 470\Omega$")
plt.errorbar(r2d[:5,0],r2d[:5,1],r2e[:5,1],r2e[:5,0], fmt = ".", label = "$R = 1000\Omega$")
plt.errorbar(r3d[:4,0],r3d[:4,1],r3e[:4,1],r3e[:4,0], fmt = ".", label = "$R = 10000\Omega$")



x1 = np.arange(-0.001,max(r1d[:4,0])+ 0.002, 0.001 )

x2 = np.arange(-0.002,max(r2d[:4,0])+ 0.005, 0.001 )

x3 = np.arange(-0.002,max(r3d[:4,0])+ 0.005, 0.001 )
#print(x)
val, err, chisq, dof, chindof = linfitr(r1d, r1e)
lin1 = lin(x1, val["m"], val["b"])
val, err, chisq, dof, chindof = linfitr(r2d, r2e)
lin2 = lin(x2, val["m"], val["b"])
#print(lin2)
val, err, chisq, dof, chindof = linfitr(r3d, r3e)
lin3 = lin(x3, val["m"], val["b"])

plt.plot(x1, lin1, label = "$Fit, 470\Omega$", color = "b", linestyle = "--", alpha =.5)
plt.plot(x2,lin2, label = "$Fit, 1000\Omega$", color = "orange", linestyle = "--", alpha =.5)
plt.plot(x3, lin3, label = "$Fit, 10000\Omega$", color = "g", linestyle = "--", alpha =.5)

plt.title("Spannungsmessung nahe Nullpunkt")
plt.errorbar(0,0,adc_err_u, adc_err_u,  label = "Nullpunkt, Digitalisierungsfehler", marker = ".", color = "black")
plt.ylabel("$U_R [V]$")
plt.xlabel("$U_{Referenzwiderstand} [V]$")

plt.legend(fontsize = 7)
plt.savefig("R_sys_null.pdf")
plt.show()

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

def linfit_z(UE, UA, UEerr, UAerr):
    'linear fitting function for Q = Ue / Ua with Zdiode'
    

    
    chi2_r = lambda m, b: chi2(lin ,y = UE, x = UA, yerr=UAerr, xerr = UEerr, a = m , c=b)

    m0 = iminuit.Minuit(chi2_r, m = 1, b = 0)

    m0.migrad()
    m0.hesse()
    return m0.values, m0.errors, m0.fval, len(UA) - 2, m0.fval/(len(UA) - 2)

def z_diode_plot(zdata, zerr, name ="", filename = "test.pdf"):
    fig, ax = fig, ax = plt.subplots(3, 1, figsize=(10,7), layout = "tight",gridspec_kw={'height_ratios': [5, 5, 2]})
    
    UE = zdata[:,0] + zdata[:,1]
    UEerr = np.sqrt(zerr[:,0]**2 + zerr[:,1]**2)

    ax[0].errorbar(UE, zdata[:,0], zerr[:,0], UEerr, fmt =".", label = "Messdaten")
    
    arbeitspunkt = -3.25
    
    ax[0].axhline(arbeitspunkt, color = "black", linestyle = "--", label = "Arbeitspunkt, $U_A$ = $-3.25V$")
    ax[0].set_xlabel("$U_E [V]$")
    ax[0].set_ylabel("$U_A [V]$")
    ax[0].legend(fontsize = 13)
    ax[0].title.set_text("Spannungsstabilisierung Z-Diode über R = " + name)

    
    
    
    
    UA_sliced = zdata[:,0][np.abs(zdata[:,0]-arbeitspunkt)<(-1/min(UE)*0.08)]
    UAerr_sliced = zerr[:,0][np.abs(zdata[:,0]-arbeitspunkt)<(-1/min(UE)*0.08)]
    UE_sliced = UE[np.abs(zdata[:,0]-arbeitspunkt)<(-1/min(UE)*0.08)]
    UEerr_sliced = UEerr[np.abs(zdata[:,0]-arbeitspunkt)<(-1/min(UE)*0.08)]
    
    
    val, err, chisq, dof, chisqdof = linfit_z(UE_sliced, UA_sliced, UEerr_sliced, UAerr_sliced)
    
    
    fity = 1/val["m"]*(UE_sliced - val["b"])
    ax[1].errorbar(UE_sliced, UA_sliced, UAerr_sliced, UEerr_sliced, fmt =".", label = "Messungen")
    ax[1].plot(UE_sliced, fity, label = "Linearer Fit")
    ax[1].axhline(arbeitspunkt, color = "black", linestyle = "--", label = "Arbeitspunkt")
    ax[1].set_xlabel("$U_E [V]$")
    ax[1].set_ylabel("$U_A [V]$")
    ax[1].legend(fontsize = 13)
    
    
    
    
    sigmaRes = np.sqrt(1/val["m"]*UEerr_sliced**2 + UAerr_sliced**2)
    
    ax[2].axhline(y=0., color='black', linestyle='--', zorder = 4)
    
    ax[2].errorbar(UE_sliced, fity-UA_sliced, sigmaRes, fmt = ".",   label = "Residuen" )
    #ax[1].fill_between(unp.nominal_values(i), fity-fityminus, fity-fityplus, alpha=0, linewidth = 0, label = "err_fit", color = "r")
    #ax[1].axhline(y=0., color='black', linestyle='--')
    ax[2].set_ylabel('$U_A- 1/G*U_E + b$ [$V$] ')
    ax[2].set_xlabel('$U_E$ [$V$] ')
    ymax = max([abs(x) for x in ax[2].get_ylim()])
    ax[2].set_ylim(-ymax, ymax)
    ax[2].legend(fontsize = 13)
    
    S = val["m"]* np.mean(UA_sliced)/np.mean(UE_sliced)
    fig.text(0.5,0, f'G = {val["m"]:.1f} , S = {S:.1f}, chi2/dof = {chisq:.1f} / {dof} = {chisqdof:.3f} ', horizontalalignment = "center")
    #fig.subplots_adjust(hspace=0.0)
    
    
    #print(f'G = ({val["m"]})')
    #print(S)
    #print(chisq)
    #print(dof)
    plt.savefig(filename)
    plt.show()
    
    return True

z_diode_100,uz_d_err_100 = read_data("z_diode_100.txt", plot_raw= True, plot_errorbar=True)
z_diode_100[:,0] = -z_diode_100[:,0]
z_diode_1000,uz_d_err_1000 = read_data("z_diode_1000.txt", plot_raw= True, plot_errorbar=True)
z_diode_1000[:,0] = -z_diode_1000[:,0]
#plt.plot(z_diode_1000[:,0],z_diode_1000[:,1])
#plt.plot(z_diode_100[:,0],z_diode_100[:,1])

z_diode_plot(z_diode_100, uz_d_err_100, "100$\Omega$", "Z_100R.pdf")
z_diode_plot(z_diode_1000, uz_d_err_1000, "1000$\Omega$", "Z_1000R.pdf")

#%%kondensator

def exp(x, a,b,c):
    return a*np.exp(-x/b )+ c
#def gaussian(x, mu, sig):
#    return (
#        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
#    )


#def expandnoise(x, a,b,c, d,e):
#    return exp(x,a,b,c) + gaussian(x, d,e)
#Data = np.genfromtxt("c_entladekurve.txt")
Data = np.genfromtxt("C_entladekurve.txt")



# zwischen 5000 und 455050
t = Data[4966:450000,0] - Data[4966,0]
U_C = Data[4966:450000,2]
U_R = Data[4966:450000,1]

U_test = Data[:,1]
diff = U_test[1:] -U_test[:-1]
plt.hist(diff)
plt.show()
print(min(abs(diff[diff != 0.0])))
#%% normal exp fit
lstsq = cost.LeastSquares(t, U_C, adc_err_u, exp)

model = Minuit(lstsq, a=10, b=1000, c=0,  )

lstsq2 = cost.LeastSquares(t, U_R, adc_err_u, exp)

model2 = Minuit(lstsq2, a=-10, b=1000, c=0,  )

model.migrad()
model.hesse()
print(model2.migrad())
model2.hesse()
#%%'
fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,8), layout = "tight",gridspec_kw={'height_ratios': [5, 2]}, sharex = True)
fitU = exp(t, model.values["a"], model.values["b"], model.values["c"], )
fitUR = exp(t, model2.values["a"], model2.values["b"], model2.values["c"], )



ax[0].title.set_text("Entladekurve Kondensator")

ax[0].errorbar(t,U_C,adc_err_u, label = "Messwerte, $U_C$")
ax[0].errorbar(t,U_R,adc_err_u, label = "Messwerte, $U_R$")

ax[0].plot(t,fitU, label = "Fit: $U_C$ = "+f'{model.values["a"]:.2f}*' + "exp(-t/"+ f'{model.values["b"]:.2f}) + {model.values["c"]:.2f}'  )
ax[0].plot(t,fitUR, label = "Fit: $U_R$")
#ax[0].set_yscale("log")
ax[0].set_ylabel("$U [V]$")
ax[0].set_xlabel("$t [ms]$")
#plt.scatter(Data[5000:455050,0],Data[5000:455050,2], )
ax[0].legend(fontsize = 13)


ax[1].errorbar(t,U_C - fitU, adc_err_u , label = "Residuen, $U_C$")
ax[1].errorbar(t,U_R - fitUR, adc_err_u , label = "Residuen, $U_R$")

ax[1].axhline(0, color = "black", linestyle = "--",)

 
              
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].set_ylabel("$U - U_{Fit} [V]$")

ax[1].set_xlabel("$t [ms]")



ax[1].legend(fontsize = 13)
#ax[1].scatter(t,U_R - fitU_R, label = "Residuen, U_R")

#ax[1].plot(t, exp(t, abs( U_C[0] - fitU[0])-abs( U_C[-1] - fitU[-1]),  m1.values["b"], abs( U_C[-1] - fitU[-1])))
#plt.legend()
plt.savefig("kondensator_entlade_fit.png", dpi = 400)
plt.show()
#%%

Data = np.genfromtxt("C_ladekurve.txt")



# zwischen 4853 und 455050
t = Data[4853:450000,0] - Data[10000,0]
U_C = Data[4853:450000,2]
U_R = Data[4853:450000,1]
#%%
lstsq = cost.LeastSquares(t, U_C, adc_err_u, exp)

model = Minuit(lstsq, a=10, b=900, c=0,  )

lstsq2 = cost.LeastSquares(t, U_R, adc_err_u, exp)

model2 = Minuit(lstsq2, a=-10, b=900, c=0,  )

model.migrad()
model.hesse()
print(model2.migrad())
model2.hesse()

fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,8), layout = "tight",gridspec_kw={'height_ratios': [5, 2]}, sharex = True)
fitU = exp(t, model.values["a"], model.values["b"], model.values["c"], )
fitUR = exp(t, model2.values["a"], model2.values["b"], model2.values["c"], )



ax[0].title.set_text("Ladekurve Kondensator")

ax[0].errorbar(t,U_C,adc_err_u, label = "Messwerte, $U_C$")
ax[0].errorbar(t,U_R,adc_err_u, label = "Messwerte, $U_R$")

ax[0].plot(t,fitU, label = "Fit: $U_C$ = "+f'{model.values["a"]:.2f}*' + "exp(-t/"+ f'{model.values["b"]:.2f}) + {model.values["c"]:.2f}'  )
ax[0].plot(t,fitUR, label = "Fit: $U_R$")
#ax[0].set_yscale("log")
ax[0].set_ylabel("$U [V]$")
ax[0].set_xlabel("$t [ms]$")
#plt.scatter(Data[5000:455050,0],Data[5000:455050,2], )
ax[0].legend(fontsize = 13)


ax[1].errorbar(t,U_C - fitU, adc_err_u , label = "Residuen, $U_C$")
ax[1].errorbar(t,U_R - fitUR, adc_err_u , label = "Residuen, $U_R$")

ax[1].axhline(0, color = "black", linestyle = "--",)

 
              
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].set_ylabel("$U_C - U_{Fit} [V]$")

ax[1].set_xlabel("$t [ms]")



ax[1].legend(fontsize = 13)
plt.savefig("kondensator_lade_fit.png", dpi = 400)
plt.show()

#print(model.fval)
#print(len(t)- 3)
#print(model.fval/(len(t)-3))
#%%ANDERER FIT

def exp_diel_abs(x, a,b , a2,b2, c):
    return a*np.exp(-x/b)+ a2*np.exp(-x/b2)+ c




Data = np.genfromtxt("C_entladekurve.txt")



# zwischen 5000 und 455050
t = Data[4966:450000,0] - Data[4966,0]
U_C = Data[4966:450000,2]
U_R = Data[4966:450000,1]
#%%

#print(t[5])

lstsq = cost.LeastSquares(t, U_C, adc_err_u, exp_diel_abs)
#lstsq_R = cost.LeastSquares(t, U_R, adc_err_u, exp)

m1 = Minuit(lstsq, a=10, b=900, c=0, a2=-1, b2 = 10000,  )
#m2 = Minuit(lstsq_R, a=-10, b=1, c=10)

print(m1.migrad())
print(m1.hesse())
#print(m2.migrad())
#print(m2.hesse())
#%%

fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,7), layout = "tight",gridspec_kw={'height_ratios': [5, 2]})

fitU = exp_diel_abs(t, m1.values["a"], m1.values["b"],m1.values["a2"], m1.values["b2"], m1.values["c"] )
#fitU = exp(t, U_C[0], 1000, 0)
#fitU_R = exp(t, m2.values["a"], m2.values["b"], m2.values["c"] )


ax[0].title.set_text("Entladekurve Kondensator mit Leckstrom/Dielektrischer Absorption")

ax[0].errorbar(t,U_C,adc_err_u, label = "Messwerte für $U_C$", zorder = 2)
ax[0].plot(t,fitU, label = "Fit für $U_C$", zorder = 1)

ax[0].set_ylabel("$U_C [V]$")

ax[0].set_xlabel("$t [ms]$")
#ax[0].set_yscale("log")
#plt.scatter(Data[5000:455050,0],Data[5000:455050,2], )
ax[0].legend(fontsize = 13)
ax[1].errorbar(t,U_C - fitU,adc_err_u, label = "Residuen,$ U_C$", zorder = 1)
#ax[1].scatter(t,U_R - fitU_R, label = "Residuen, U_R")
ax[1].axhline(0, color = "black", linestyle = "--",zorder  =2)

 
              
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].set_ylabel("$U_C - U_{Fit} [V]$")

ax[1].set_xlabel("$t [ms]$")

#ax[1].plot(t, exp(t, abs( U_C[0] - fitU[0])-abs( U_C[-1] - fitU[-1]),  m1.values["b"], abs( U_C[-1] - fitU[-1])))
ax[1].legend(fontsize = 13)
#plt.legend()
plt.savefig("c_entlade_leckstrom.png", dpi = 400)
plt.show()
#print(m1.fval)
#print(len(t)- 3)
#print(m1.fval/(len(t)-3))
 #%%gleichrichter
wechselspannung = np.genfromtxt("Wechselspannung_kein_R_L.txt")
plt.errorbar(wechselspannung[:7000,0], wechselspannung[:7000,1], adc_err_u, fmt = ".", label= "Messwerte, $V_{SS} = 4V, f = 50Hz$")
plt.title("Wechselspannung ohne Last für Gleichrichter")
plt.axhline(-2, color = "black", linestyle = "--")
plt.axhline(2, color = "black", linestyle = "--")
plt.xlabel("$t [ms]$")
plt.ylabel("$U_{In} [V]$")
plt.legend(loc = "upper right")
plt.savefig("Wechselspannung_Gleichrichter.pdf")
plt.show()


gleichrichter = np.genfromtxt("Gleichrichter_100R.txt")
plt.title("Gleichrichter, $R_L = 100\Omega$")
plt.errorbar(gleichrichter[:6000,0], gleichrichter[:6000,2], adc_err_u, fmt = ".", label = "$U_1$")


plt.errorbar(gleichrichter[:6000,0], gleichrichter[:6000,1], adc_err_u, fmt = ".", label = "$U_2$")


#ueff_berechnet = np.sqrt((gleichrichter[-1,0]-gleichrichter[0,0]) * )
ueff_gemessen = np.sqrt(np.sum(gleichrichter[:, 2]**2 )/len(gleichrichter[:, 2]))

ueff_theoretisch = max(abs(gleichrichter[:, 2])) / np.sqrt(2)

plt.axhline(ueff_gemessen, color = "black", linestyle = "--", label = "$U_{Eff}$" + f" = {ueff_gemessen:.2f}V, gemessen")
plt.axhline(ueff_theoretisch, color = "grey", linestyle = "--", label = "$U_{Eff}$" + f" = {ueff_theoretisch:.2f}V, theoretisch")

#plt.scatter(gleichrichter[:4000,0], gleichrichter[:4000,1] + gleichrichter[:4000,2], marker = ".", color = "red", label  ="U3")
plt.xlabel("$t [ms]$")
plt.ylabel("$U [V]$")
plt.legend(loc = "lower right", fontsize = 10)
plt.savefig("gleichrichter100R.pdf")
plt.show()

######

gleichrichter = np.genfromtxt("Gleichrichter_1000R_Neu.txt")
plt.errorbar(gleichrichter[:6000,0], gleichrichter[:6000,2], adc_err_u, fmt = ".", label = "$U_1$")
plt.title("Gleichrichter, $R_L = 1000\Omega$")

#offset = (max(gleichrichter[:4000,1] + gleichrichter[:4000,2]) + min(gleichrichter[:1200,1] + gleichrichter[:1200,2]))/2
plt.errorbar(gleichrichter[:6000,0], gleichrichter[:6000,1], adc_err_u, fmt = ".", label = "$U_2$")
#plt.scatter(gleichrichter[:4000,0], gleichrichter[:4000,1] + gleichrichter[:4000,2], marker = ".", color = "red", label  ="U3")

ueff_gemessen = np.sqrt(np.sum(gleichrichter[:, 2]**2 )/len(gleichrichter[:, 2]))

ueff_theoretisch = max(abs(gleichrichter[:, 2])) / np.sqrt(2)

plt.axhline(ueff_gemessen, color = "black", linestyle = "--", label = "$U_{Eff}$" + f" = {ueff_gemessen:.2f}V, gemessen")
plt.axhline(ueff_theoretisch, color = "grey", linestyle = "--", label = "$U_{Eff}$" +f" = {ueff_theoretisch:.2f}V, theoretisch")
#print(np.sqrt(np.sum(wechselspannung[:, 1]**2 )/len(wechselspannung[:, 1])))


plt.xlabel("$t [ms]$")
plt.ylabel("$U [V]$")
plt.legend(loc = "lower right", fontsize = 10)
plt.savefig("gleichrichter1000R.pdf")

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

#read_data()

# plt.scatter(u100mv[:,0],u100mv[:,1])
# plt.scatter(u600mv[:,0],u600mv[:,1])
# plt.scatter(u700mv[:,0],u700mv[:,1])
# plt.scatter(u800mv[:,0],u800mv[:,1])
# plt.plot()
#%%schmitttrigger
schmitt_hysterese = np.genfromtxt("schmitt_trigger_hysterese.txt")
print(max(schmitt_hysterese[:,2]))
trigger_active = schmitt_hysterese[:,0][schmitt_hysterese[:,2] > 6]


plt.scatter(schmitt_hysterese[:,0], schmitt_hysterese[:,1], marker = ".")
plt.scatter(schmitt_hysterese[:,0], schmitt_hysterese[:,2], marker = ".")
plt.fill_between(trigger_active, max(schmitt_hysterese[:,2]), alpha = 0.5, color = "orange")

#plt.plot(schmitt_hysterese[:,0], schmitt_hysterese[:,2], color = "r")

plt.axhline(2.95)
plt.axhline(1.65)



plt.show()


