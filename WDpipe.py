import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import optimize
import wd_fit_tools
from scipy import interpolate
import pickle
from scipy import linalg
import wd_fit_tools
import pandas as pd
import lmfit
plot=True
wave_type="a" #change this to "v" if spectrum is in vacuum wavelength
c = 299792.458 # Speed of light in km/s
infile=sys.argv[1]
wave_in,flux,err=np.loadtxt(infile,usecols=(0,1,2),delimiter=",",unpack=True)

flux=flux[err!=0.]
wave_in=wave_in[err!=0.]
err=err[err!=0.]
wave_in=wave_in[(np.isnan(flux)==False)]
err=err[(np.isnan(flux)==False)]
flux=flux[(np.isnan(flux)==False)]
WDclass, prob= wd_fit_tools.WDclassify(wave_in,flux,err)
#these are arbitrary optical spectra cuts
err=err[(wave_in>=3500)&(wave_in<9000)] 
flux=flux[(wave_in>=3500)&(wave_in<9000)]
wave_in=wave_in[(wave_in>=3500)&(wave_in<9000)]
#try:
 #   wave_type=sys.argv[2]
  #  if wave_type!="a" or wave_type!="v":
   #     print("specify air a, or vacuum v wavelength, air wavelength assumed")
    #    wave_type="a"
#except:
 #   print("specify air a, or vacuum v wavelength, air wavelength assumed")
  #  wave_type="a"

try:
    parallax=float(sys.argv[2])
    if parallax==0:
        parallax=0.00001 #0 parallax breaks things
except:
    parallax=None
try:
    Gaia_G_mag=float(sys.argv[3])    
except:
    Gaia_G_mag=None
if Gaia_G_mag!=None and parallax!=None:
    try:
        GaiaID=sys.argv[4]
    except:
        GaiaID=None
else:
    try:
        GaiaID=sys.argv[2]
    except:
        GaiaID=None
if wave_type=="a":
    wave=wave_in/(1.0 + 2.735182e-4 + 131.4182/wave_in**2 + 2.76249e8/wave_in**4) #models are in vacuum wavelength so this conversion may be needed if spectrum is in air wavelenght
elif wave_type=="v":
    wave=wave_in
    
print("Spectrum classified as:",WDclass, prob)
if WDclass!="DA":
    print("Object is not a DA. Stopping here")
else:
    spectra=np.stack((wave,flux,err),axis=-1)
    spec_w=wave
    SNR=np.median(flux[(wave<5200)&(wave>3800)]/err[(wave<5200)&(wave>3800)])
    blue_SNR=np.mean(flux[(wave<4050)&(wave>3800)]/err[(wave<4050)&(wave>3800)])
    H_SNR=np.mean(flux[(wave<6684)&(wave>6444)]/err[(wave<6684)&(wave>6444)])
    if blue_SNR >15:
        m_line=5
        No=1
    else:
        m_line=5
        No=1
    if H_SNR >8:
        s_line=0
    else:
        s_line=1

#Initial log g, Teff for fitting
    tl= pd.read_csv('reference_phot_tlogg.csv') #load reference table with parameters from Gentile Fusillo et al. 2021
    sourceID=np.array(tl[u'source_id']).astype(str)
    T_H=np.array(tl[u'teff_H']).astype(float)
    log_H=np.array(tl[u'logg_H']).astype(float)
    eT_H=np.array(tl[u'eteff_H']).astype(float)
    elog_H=np.array(tl[u'elogg_H']).astype(float)
    spec_n, cont_flux = wd_fit_tools.norm_spectra(spectra,mod=False)#,g_line_crop,mod=False)
    if (GaiaID!= None) and (GaiaID in sourceID):
        first_T=T_H[sourceID==GaiaID][0]
        first_g=log_H[sourceID==GaiaID][0]*100
        initial=1
        #load lines to fit and crops them
    else:
        g_line_crop = np.loadtxt('line_crop.dat')
        g_l_crop = g_line_crop[(g_line_crop[:,0]>spec_w.min()) & (g_line_crop[:,1]<spec_w.max())]
        spec_n, cont_flux = wd_fit_tools.norm_spectra(spectra,mod=False)#,g_line_crop,mod=False)
        #fit entire grid to find good starting point
        lines_sp,lines_mod,best_grid,grid_param,grid_chi=wd_fit_tools.fit_grid(spec_n,g_line_crop)
        first_T=grid_param[grid_chi==np.min(grid_chi)][0][0]
        first_g=800#grid_param[grid_chi==np.min(grid_chi)][0][1]
        initial=0

#Based on Teff determine what line crop to use
    if first_T > 80000:
        first_T=80000
    if first_g < 701:
        first_g=701
    if first_T>=16000 and first_T<=40000:
        line_crop = np.loadtxt('line_crop.dat',skiprows=1)#,max_rows=6)#,skiprows=s_line)
    elif first_T>=8000 and first_T<16000:
        line_crop = np.loadtxt('line_crop_cool.dat',skiprows=1)#,max_rows=m_line)#,skiprows=s_line)
    elif first_T<8000:
        line_crop = np.loadtxt('line_crop_vcool.dat',skiprows=1)#,max_rows=m_line)#,skiprows=s_line)
    elif first_T>40000:
        line_crop = np.loadtxt('line_crop_hot.dat',skiprows=1)#,max_rows=m_line)#,skiprows=s_line)
    l_crop = line_crop[(line_crop[:,0]>spec_w.min()) & (line_crop[:,1]<spec_w.max())]
    
#-----------------------load PCA model for precise fit------------------------------------------------------
    wref=np.load("wref_4most.npy") #this is the models reference wavelength
    with open("emu_file_4most.mod", 'rb') as pickle_file:
        emu= pickle.load(pickle_file)
#-----------------------------------------------------------------------------------------
#set parameters to use in the fit
    fit_params = lmfit.Parameters()
    fit_params['teff'] = lmfit.Parameter(name="teff",value=first_T,min=3000,max=80000)
    fit_params['logg'] = lmfit.Parameter(name="logg",value=first_g,min=701,max=949)
    fit_params['rv'] = lmfit.Parameter(name="rv",value=0.2, min=-80, max=80)
#normalize the croppped lines. This seems to introduce less problems than normalizing the full spectrum
    spec_nl=wd_fit_tools.da_line_normalize(spectra,l_crop,mod=False)
    new_best= lmfit.minimize(wd_fit_tools.line_func_rv,fit_params,args=(spec_nl,l_crop,emu,wref),method="leastsq") #perform the fit

#======== In SDSS some clustering was noticed around specific logg values. This is a hack to try and solve it. Not necessary for now
    #prob_list=[7.01,9.49,7.5,8.2,8.]
    #if round(new_best.params['logg'].value/100,4) in prob_list:
    #    
    #    if first_T>=16000 and first_T<=40000:
    #        line_crop = np.loadtxt('line_crop.dat',max_rows=6)
    #    elif first_T>=8000 and first_T<16000:
    #        line_crop = np.loadtxt('line_crop_cool.dat',max_rows=6)
    #    elif first_T<8000:
    #        line_crop = np.loadtxt('line_crop_vcool.dat',max_rows=3)
    #    elif first_T>40000:
    #        line_crop = np.loadtxt('line_crop_hot.dat',max_rows=6)
    #    l_crop = line_crop[(line_crop[:,0]>spec_w.min()) & (line_crop[:,1]<spec_w.max())]
    #    new_best= lmfit.minimize(wd_fit_tools.line_func_rv,fit_params,args=(spec_nl,l_crop,emu,wref),method="leastsq")
#==============================================================================================================================
    best_T=new_best.params['teff'].value
    best_Te=new_best.params['teff'].stderr
    if best_Te==None:
        best_Te=0.0
    best_g=new_best.params['logg'].value
    best_ge=new_best.params['logg'].stderr
    best_rv=new_best.params['rv'].value
    if best_ge==None:
        best_ge=0.0
    chi2=new_best.redchi
  
    if initial==1: # if Teff and logg from reference table were used, this should be the only solution
        if best_ge!=None:
            print("ONLY Solution: Teff=",best_T,"+-",best_Te," log g=",best_g,"+-",best_ge)

    elif initial==0: # otherwise need to calculate hot/cold solution
        fit_params = lmfit.Parameters()
        fit_params['logg'] = lmfit.Parameter(name="logg",value=800,min=701,max=949)
        fit_params['rv'] = lmfit.Parameter(name="rv",value=0.2, min=-80, max=80)
        
        if first_T <=13000.:
            tmp_Tg,tmp_chi= grid_param[grid_param[:,0]>13000.], grid_chi[grid_param[:,0]>13000.]
            second_T= tmp_Tg[tmp_chi==np.min(tmp_chi)][0][0]
            fit_params['teff'] = lmfit.Parameter(name="teff",value=second_T,min=13000,max=80000)

        elif first_T >13000.:
            tmp_Tg,tmp_chi= grid_param[grid_param[:,0]<13000.], grid_chi[grid_param[:,0]<13000.]
            second_T= tmp_Tg[tmp_chi==np.min(tmp_chi)][0][0]
            fit_params['teff'] = lmfit.Parameter(name="teff",value=second_T,min=3000,max=13000)

        second_g=800
        if second_T>=16000 and second_T<=40000:
            line_crop = np.loadtxt('line_crop.dat',skiprows=1)#,max_rows=m_line,skiprows=s_line)
        elif second_T>=8000 and second_T<16000:
            line_crop = np.loadtxt('line_crop_cool.dat',skiprows=1)#,max_rows=m_line,skiprows=s_line)
        elif second_T<8000:
            line_crop = np.loadtxt('line_crop_vcool.dat',skiprows=1)#,max_rows=m_line,skiprows=s_line)
        elif second_T>40000:
            line_crop = np.loadtxt('line_crop_hot.dat',skiprows=1)#,max_rows=m_line,skiprows=s_line)
        l_crop = line_crop[(line_crop[:,0]>spec_w.min()) & (line_crop[:,1]<spec_w.max())]
        
# repeat fit for second starting point
        second_best= lmfit.minimize(wd_fit_tools.line_func_rv,fit_params,args=(spec_nl,l_crop,emu,wref),method="leastsq")

        best_T2=second_best.params['teff'].value
        best_Te2=second_best.params['teff'].stderr
        if best_Te2==None:
            best_Te2=0.0
        best_g2=second_best.params['logg'].value
        best_ge2=second_best.params['logg'].stderr
        if best_ge2==None:
            best_ge2=0.0
        if Gaia_G_mag!= None and parallax != None:
#========================use gaia G mag and parallax to solve for hot vs cold solution
            T_true=wd_fit_tools.hot_vs_cold(best_T,best_g/100,best_T2,best_g2/100,parallax,Gaia_G_mag,emu,wref)
            if T_true==best_T:
                print("FIRST Solution: Teff=",best_T,"+-",best_Te," log g=",best_g,"+-",best_ge)#,"+-",perr[2])
            elif T_true==best_T2:
                print("SECOND Solution: Teff=",best_T2,"+-",best_Te2," log g=",best_g2,"+-",best_ge2)#,"+-",perr2[2])
   
        else: # if Gaia_G_mag and parallax are not available return both solutions
            print("FIRST Solution: Teff=",best_T,"+-",best_Te," log g=",best_g,"+-",best_ge)#,"+-",perr[2])
            print("SECOND Solution: Teff=",best_T2,"+-",best_Te2," log g=",best_g2,"+-",best_ge2)#,"+-",perr2[2])
if plot and WDclass=="DA":
    lines_s,lines_m,mod_n=wd_fit_tools.fit_func((best_T,best_g,best_rv), spec_n,l_crop,emu,wref,mode=1)
    if initial==0:
        lines_s_o,lines_m_o,mod_n_o=wd_fit_tools.fit_func((best_T2,best_g2,best_rv),spec_n,l_crop,emu,wref,mode=1)
    fig=plt.figure(figsize=(8,5))
    ax1 = plt.subplot2grid((1,4), (0, 3),rowspan=3)
    step = 0
    for i in range(0,len(lines_s)): # plots Halpha (i=0) to H6 (i=5)
        min_p   = lines_s[i][:,0][lines_s[i][:,1]==np.min(lines_s[i][:,1])][0]
        ax1.plot(lines_s[i][:,0]-min_p,lines_s[i][:,1]+step,color='k')
        ax1.plot(lines_s[i][:,0]-min_p,lines_m[i]+step,color='r')
        if initial==0:
            min_p_o = lines_s_o[i][:,0][lines_s_o[i][:,1]==np.min(lines_s_o[i][:,1])][0]
            ax1.plot(lines_s_o[i][:,0]-min_p_o,lines_m_o[i]+step,color='g')
        step+=0.5
    xticks = ax1.xaxis.get_major_ticks()
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax2 = plt.subplot2grid((3,4), (0, 0),colspan=3,rowspan=3)
    full_spec=np.stack((wave,flux,err),axis=-1)
    full_spec = full_spec[(np.isnan(full_spec[:,1])==False) & (full_spec[:,0]>3500)& (full_spec[:,0]<7900)]
        

    # Adjust the flux of models to match the spectrum
    check_f_spec=full_spec[:,1][(full_spec[:,0]>4500.) & (full_spec[:,0]<4550.)]
    check_f_model=mod_n[:,1][(mod_n[:,0]>4500.) & (mod_n[:,0]<4550.)]
    adjust=np.average(check_f_model)/np.average(check_f_spec)
    ax2.plot(full_spec[:,0],full_spec[:,1],color='k')
    ax2.plot(mod_n[:,0]*(best_rv+c)/c,(mod_n[:,1]/adjust),color='r')
    if initial==0:
        check_f_model_o=mod_n_o[:,1][(mod_n_o[:,0]>4500.) & (mod_n_o[:,0]<4550.)]
        adjust_o=np.average(check_f_model_o)/np.average(check_f_spec)
        ax2.plot(mod_n_o[:,0]*(best_rv+c)/c,mod_n_o[:,1]/adjust_o,color='g')

    ax2.set_ylabel(r'F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1} \AA^{-1}$]',fontsize=12)
    ax2.set_xlabel(r'Wavelength $(\AA)$',fontsize=12)
    ax2.set_xlim([3400,5600])
    #ax3 = plt.subplot2grid((3,4), (2, 0),colspan=3,rowspan=1,sharex=ax2)

    #flux_i = interpolate.interp1d(mod_n[:,0]*(best_rv+c)/c,mod_n[:,1]/adjust,kind='linear')(full_spec[:,0])
    #wave3=full_spec[:,0]
    #flux3=full_spec[:,1]/flux_i
    #binsize=1
    #xdata3=[]
    #ydata3=[]
    #for i in range(0,(np.size(wave3)-binsize),binsize):
     #   xdata3.append(np.average(wave3[i:i+binsize]))
     #   ydata3.append(np.average(flux3[i:i+binsize]))
    #plt.plot(xdata3,ydata3)
    #plt.hlines(1.02, 3400,5600,colors="r")
    #plt.hlines(1.01, 3400,5600,colors="0.5",ls="--")
    #plt.hlines(0.98, 3400,5600,colors="r")
    #plt.hlines(0.99, 3400,5600,colors="0.5",ls="--")
    #ax3.set_xlim([3400,5600])
    #ax3.set_ylim([0.95,1.04])
    plt.show()
elif plot:
    ax = plt.subplot(111)
    full_spec=np.stack((wave,flux,err),axis=-1)
    full_spec = full_spec[(np.isnan(full_spec[:,1])==False) & (full_spec[:,0]>3500)& (full_spec[:,0]<9000)]
    ax.plot(full_spec[:,0],full_spec[:,1],color='k')
    ax.set_ylabel(r'F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1} \AA^{-1}$]',fontsize=12)
    ax.set_xlabel(r'Wavelength $(\AA)$',fontsize=12)
    ax.set_xlim([3400,8000])
    plt.show()
