import model_processing
#import emulator_DA
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
import bisect
from scipy.interpolate import splrep,splev
import scipy.interpolate
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.optimize import curve_fit
import pickle
h=6.626e-34
c=2.997e18
k=1.38e-23
def da_line_normalize(spectra,l_crop,mod=True): #individual Balmer line normalization
    import matplotlib.pyplot as plt
    sn_w=spectra[:,0]
    cropped_w=[]
    cropped_f=[]
    cropped_e=[]
    for i in range(len(l_crop)):
        
        if not mod:
            l_c0,l_c1 = l_crop[i,0],l_crop[i,1]
            line = spectra[(sn_w>=l_c0)&(sn_w<=l_c1)]
            init_f=np.mean(line[:,1][0:15])
            end_f=np.mean(line[:,1][-15:])
            init_w=np.mean(line[:,0][0:15])
            end_w=np.mean(line[:,0][-15:])
  
        else:
            l_c0,l_c1 = l_crop[i,0]-10,l_crop[i,1]+10
            line = spectra[(sn_w>=l_c0)&(sn_w<=l_c1)]
            init_f=np.max(line[:,1][0:30])
            end_f=np.max(line[:,1][-30:])
            init_w=line[:,0][line[:,1]==init_f][0]
            end_w=line[:,0][line[:,1]==end_f][0]
        z=np.polyfit([init_w,end_w], [init_f,end_f],1)
        lin = np.poly1d(z)
        cont=lin(line[:,0])
        if not mod:
            n_err=line[:,2]/cont
            cropped_e.append(n_err)
        n_flux=line[:,1]/cont
        cropped_f.append(n_flux)
        cropped_w.append(line[:,0])
        #plt.plot(line[:,0],line[:,1])
        #plt.plot(line[:,0],cont,c="r")
        #plt.scatter([init_w,end_w], [init_f,end_f],c="r")
        #plt.show()
    all_cropped_w=np.concatenate((cropped_w),axis=0)
    all_cropped_f=np.concatenate((cropped_f),axis=0)
    if not mod:
        all_cropped_e=np.concatenate((cropped_e),axis=0)
        norm_lines=np.stack((all_cropped_w,all_cropped_f,all_cropped_e),axis=-1)
    else:
        norm_lines=np.stack((all_cropped_w,all_cropped_f),axis=-1)
    #plt.plot(norm_lines[:,0],norm_lines[:,1])
    #plt.show()
    return norm_lines


def line_func_rv(params,_sn, _l,emu,wref):
    import lmfit
    parvals = params.valuesdict()
    _T = parvals['teff']
    _g = parvals['logg']
    _rv = parvals['rv']
    #print(_T,_g,_rv)
    recovered=generate_modelDA(_T,(_g),emu)
    model=np.stack((wref,recovered),axis=-1)
    model=convolve_gaussian_R(model,1700)
    model[:,0]=model[:,0]+_rv
    _l=_l#+_rv
    norm_model=da_line_normalize(model,_l)
    m_wave_n, m_flux_n = norm_model[:,0], norm_model[:,1]
    sn_w=_sn[:,0]
    lines_m, lines_s, sum_l_chi2 = [],[],0
    flux_s,err_s=[],[]
    chi_line=[]
    for i in range(len(_l)):

        _l_c=_l
        m_wave_n_c=m_wave_n
             # Crop model and spec to line
        l_c0,l_c1 = _l_c[i,0],_l_c[i,1]
        l_m= m_flux_n[(m_wave_n>=l_c0-1)&(m_wave_n<=l_c1+1)]
        l_m_w=m_wave_n[(m_wave_n>=l_c0-1)&(m_wave_n<=l_c1+1)]
        l_s = _sn[(sn_w>=l_c0)&(sn_w<=l_c1)]
        l_m= interpolate.interp1d(l_m_w,l_m,kind='linear')(l_s[:,0])
        lines_m.append(l_m)
        lines_s.append(l_s)       
        flux_s.append(l_s[:,1])
        err_s.append(l_s[:,2])
        chi_line.append(np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2))#/np.size(l_m))
   
    all_lines_m=np.concatenate((lines_m),axis=0)
    all_lines_s=np.concatenate((flux_s),axis=0)
    all_err_s=np.concatenate((err_s),axis=0)
    sum_l_chi2=np.array(((all_lines_s-all_lines_m)/all_err_s)**2)
    chi_line=np.array(chi_line)
    chi_line_s=np.sort(chi_line)
    chi_line_m = chi_line_s[:-1]
    chi_sum=np.mean(chi_line)
    chi=np.array((all_lines_s-all_lines_m)/all_err_s)
    return(sum_l_chi2)


def norm_spectra(spectra,model=True,add_infinity=False,norm=1,mod=True):
    """
    Normalised spectra by DA  continuum regions 
    spectra of form array([wave,flux,error]) (err not necessary so works on models)
    only works on SDSS spectra region
    Optional:
        EDIT n_range_s to change whether region[j] is fitted for a peak or mean'd
        add_infinity=False : add a spline point at [inf,0]
    returns spectra, cont_flux
    """
    if mod:
        start_n=np.array([3630,3675.,3770,3805,3835.,3895.,3995.,4210,4490.,4620.,5070.,5200.,
                      5600.,6000.,7000.,7400.,7700.])
        end_n=np.array([3660,3725,3795,3830,3885.,3960.,4075.,4240,4570.,4670.,5100.,5300.,
                        5800.,6100.,7150.,7500.,7800.])
        n_range_s=np.array(['M','M','P','P','P','P','P','P','M','M','M','M','M','M','M','M','M','M','M','M'])
        

    else:
        if norm==1:
            start_n=np.array([3700,3770,3835.,3900.,3995.,4180,4490.,5070.,5200.,
                              5600.,6000.,6120,6300,6800,7000.,7400.,7700.])
            end_n=np.array([3720,3795,3885.,3950,4075.,4240,4570.,5100.,5300.,
                            5800.,6100.,6200,6340,6850,7150.,7500.,7800.])
            n_range_s=np.array(['M','M','P','P','P','P','M','M','M','M','M','M','M','M','M','M','M','M'])

        elif norm==2: #for now not used....needs fixing
            start_n=np.array([3630,3675.,3770,3815,3855.,3900,4025.,4200,4490.,4620.,5070.,5200.,
                      5600.,6000.,7000.,7400.,7700.])
            end_n=np.array([3660,3725,3795,3820,3865.,3960,4035.,4220,4570.,4670.,5100.,5300.,
                    5800.,6100.,7150.,7500.,7800.])
    
            n_range_s=np.array(['M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M'])
         
    if len(spectra[0])>2:
        snr = np.zeros([len(start_n),3])
        spectra[:,2][spectra[:,2]==0.] = spectra[:,2].max()
    else: 
        snr = np.zeros([len(start_n),2])
    wav = spectra[:,0]
    for j in range(len(start_n)):
        if (start_n[j] < wav.max()) & (end_n[j] > wav.min()):
            _s = spectra[(wav>=start_n[j])&(wav<=end_n[j])]
            _w = _s[:,0]
            #Avoids gappy spectra
            k=3 # Check if there are more points than 3
            if len(_s)>k:
                #interpolate onto 10* resolution
                l = np.linspace(_w.min(),_w.max(),(len(_s)-1)*10+1)
                if len(spectra[0])>2:
                    tck = interpolate.splrep(_w,_s[:,1],w=1/_s[:,2], s=1000)
                    #median errors for max/mid point
                    snr[j,2] = np.median(_s[:,2]) / np.sqrt(len(_w))
                else: tck = interpolate.splrep(_w,_s[:,1],s=0.0)
                f = interpolate.splev(l,tck)
                #find maxima and save
                if n_range_s[j]=='P':
                    if np.size(l[f==np.max(f)])>1:
                        if model==False:
                            snr[j,0]= np.mean(l)
                        else:
                            snr[j,0]= l[f==np.max(f)][0]
                    else:
                        if model==False:
                            snr[j,0]= np.mean(l)
                        else:
                            snr[j,0]= l[f==np.max(f)]
                    if model==False:
                        n=int(np.size(_s[:,1])/3.)
                        f_sort=np.sort(_s[:,1])
                        #errs=[np.where(f==i) for i in f_sort]
                        top5=f_sort[-n:]
                        #res = np.flatnonzero(np.isin(f, top5))  # NumPy v1.13+
                        snr[j,1]= np.average(top5)#,weights=[res])
                    else:
                        snr[j,1]= f[f==np.max(f)][0]
                #find mean and save
                elif n_range_s[j]=='M':
                    snr[j,0:2] = np.mean(l), np.mean(f)
                else: print('Unknown n_range_s, ignoring')
    snr = snr[ snr[:,0] != 0 ]
    #t parameter chosen by eye. Position of knots.
    if snr[:,0].max() < 6460: knots =[4100,4340,4500,4860,int(snr[:,0].max()-5)]
    else: knots = [3885,4340,4900,6460,7500]
    if snr[:,0].min() > 3885:
        print('Warning: knots used for spline norm unsuitable for high order fitting')
        knots=knots[1:]
    if (snr[:,0].min() > 4340) or (snr[:,0].max() < 4901): 
        knots=None # 'Warning: knots used probably bad'
   
                   
    if add_infinity: # Adds points at inf & 0 for spline to fit to err = mean(spec err)
        if snr.shape[1] > 2:
            mean_snr = np.mean(snr[:,2])
            snr = np.vstack([ snr, np.array([90000. ,0., mean_snr ]) ])
            snr = np.vstack([ snr, np.array([100000.,0., mean_snr ]) ])
        else:
            snr = np.vstack([ snr, np.array([90000.,0.]) ])
            snr = np.vstack([ snr, np.array([100000.,0.]) ])
    try: #weight by errors
        if len(spectra[0])>2:
            tck = interpolate.splrep(snr[:,0],snr[:,1], w=1/snr[:,2], t=knots, k=3)
        else: tck = interpolate.splrep(snr[:,0],snr[:,1], t=knots, k=3)
    except ValueError:
        knots=None
        if len(spectra[0])>2: 
            tck = interpolate.splrep(snr[:,0],snr[:,1], t=knots, k=3)
        else: tck = interpolate.splrep(snr[:,0],snr[:,1], t=knots, k=3)
    if mod:
        spline = interpolate.splrep(snr[:,0],snr[:,1],k=3)#,s=0.001)
    else:
        spline = interpolate.splrep(snr[:,0],snr[:,1],w=1/snr[:,2],k=3,s=3)
    cont_flux = interpolate.splev(wav,spline)
    f_ret=spectra[:,1]/cont_flux
    if mod:
        spectra_ret=np.stack((spectra[:,0],f_ret),axis=-1)
    else:
        e_ret=spectra[:,0]/cont_flux
        spectra_ret=np.stack((spectra[:,0],f_ret,e_ret),axis=-1)
       
#=======================plot for diagnostic============================
    #if not mod:
     #   import matplotlib.pyplot as plt
        #print(spectra_ret)
      #  plt.plot(spectra[:,0],spectra[:,1],zorder=1)
       # plt.plot(spectra[:,0],cont_flux,zorder=2)
        #plt.scatter(snr[:,0],snr[:,1],c="r",zorder=3)
        #plt.plot(spectra_ret[:,0],spectra_ret[:,1])
        #plt.show()
#======================================================================
    return spectra_ret, cont_flux

def fit_grid(specn,l_crop):
    #load normalised models and linearly interp models onto spectrum wave
    specn = specn[(specn[:,0]>3500)& (specn[:,0]<7500)]
    m_wave=np.arange(3000,8000,0.5)
    m_flux_n=np.load("da_flux_cube.npy")
    m_param=np.load("da_param_cube.npy")
    sn_w = specn[:,0]
    m_flux_n_i = interpolate.interp1d(m_wave,m_flux_n,kind='linear')(sn_w)
    #Crops models and spectra in a line region, renorms models, calculates chi2
    tmp_lines_m, lines_s, l_chi2 = [],[],[]
    for i in range(len(l_crop)):
        l_c0,l_c1 = l_crop[i,0],l_crop[i,1]
        l_m = m_flux_n_i.transpose()[(sn_w>=l_c0)&(sn_w<=l_c1)].transpose()
        l_s = specn[(sn_w>=l_c0)&(sn_w<=l_c1)]
        l_m = l_m*np.sum(l_s[:,1])/np.sum(l_m,axis=1).reshape([len(l_m),1])
        l_chi2.append( np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2,axis=1) )
        tmp_lines_m.append(l_m)
        lines_s.append(l_s)
    #mean chi2 over lines and stores best model lines for output
    lines_chi2, lines_m = np.sum(np.array(l_chi2),axis=0), []
    is_best = lines_chi2==lines_chi2.min()
    for i in range(len(l_crop)): lines_m.append(tmp_lines_m[i][is_best][0])
    best_TL = m_param[is_best][0]
    return  lines_s,lines_m,best_TL,m_param,lines_chi2


def generate_modelDA(teff,logg,emu):
    recovered = emu([np.log10(teff), logg])
    return(recovered)

def convolve_gaussian(spec, FWHM): #this is necessary to change resolution of the models. current models are already saved at 4MOST resolution
  """
  Convolve spectrum with a Gaussian with specifed FWHM.
  Causes wrap-around at the end of the spectrum.
  """
  sigma = FWHM/2.355
  x=spec[:,0]
  y=spec[:,1]
  def next_pow_2(N_in):
    N_out = 1
    while N_out < N_in:
      N_out *= 2
    return N_out

  #oversample data by at least factor 10 (up to 20).
  xi = np.linspace(x[0], x[-1], next_pow_2(10*len(x)))
  yi = interpolate.interp1d(x, y)(xi)

  yg = np.exp(-0.5*((xi-x[0])/sigma)**2) #half gaussian
  yg += yg[::-1]
  yg /= np.sum(yg) #Norm kernel

  yiF = np.fft.fft(yi)
  ygF = np.fft.fft(yg)
  yic = np.fft.ifft(yiF * ygF).real
  new_spec=np.stack((x,interpolate.interp1d(xi, yic)(x)),axis=-1)
  return new_spec

#

def convolve_gaussian_R(spec, R):
  """
  Similar to convolve_gaussian, but convolves to a specified 
  resolution R  rather than a specfied FWHM.
  """
  x=spec[:,0]
  y=spec[:,1]
  in_spec=np.stack((np.log(x),y),axis=-1)
  new_tmp= convolve_gaussian(in_spec, 1./R)
  new_spec=np.stack((x,new_tmp[:,1]),axis=-1)
  return new_spec


def hot_vs_cold(T1,g1,T2,g2,parallax,GaiaG,emu,wref):
    M_bol_sun, Teff_sun, Rsun_cm, R_sun_pc = 4.75, 5780., 69.5508e9, 2.2539619954370203e-08
    R1=R_from_Teff_logg(T1, g1)
    R2=R_from_Teff_logg(T2, g2)
    mod1=generate_modelDA(T1,g1*100,emu)
    flux1=(mod1/1e8)/((1000/parallax)*3.086e18)
    flux1=flux1/((1000/parallax)*3.086e18)
    flux1=flux1*(np.pi*(R1*Rsun_cm)**2)
    wave1=wref
    mod2=generate_modelDA(T2,g2*100,emu)
    #flux2=(mod2/1e8)/(((1000/parallax)*3.086e18)**2)*(np.pi*(R2*Rsun_cm)**2)
    flux2=(mod2/1e8)/((1000/parallax)*3.086e18)
    flux2=flux2/((1000/parallax)*3.086e18)
    flux2=flux2*(np.pi*(R2*Rsun_cm)**2)
    
    wave2=wref
    flux_G1,mag_G1=synthG(wave1,flux1)
    flux_G2,mag_G2=synthG(wave2,flux2)
    #print(mag_G1,mag_G2,GaiaG)
    if abs(mag_G1-GaiaG)<=abs(mag_G2-GaiaG):
        return(T1)
    else:
        return(T2)

def synthG(spectrum_w,spectrum_f):
    #spec=np.stack((spectrum_w, spectrum_f),axis=-1)
    fmin=3320.
    fmax=10828.
    filter_w,filter_r=np.loadtxt("GAIA_GAIA3.G.dat",usecols=(0,1),unpack=True)    
    ifT = np.interp(spectrum_w, filter_w,filter_r, left=0., right=0.)
    nonzero = np.where(ifT > 0)[0]
    nonzero_start = max(0, min(nonzero) - 5)
    nonzero_end = min(len(ifT), max(nonzero) + 5)
    ind = np.zeros(len(ifT), dtype=bool)
    ind[nonzero_start:nonzero_end] = True
 
    spec_flux = spectrum_f[ind]
    a = np.trapz( ifT[ind] * spec_flux*spectrum_w[ind], spectrum_w[ind], axis=-1)
    b = np.trapz( ifT[ind]*spectrum_w[ind], spectrum_w[ind])
    if (np.isinf(a).any() | np.isinf(b).any()):
        print("Warn for inf value")
    nf=a/b
    ew=5836.
    c=2.99792e10
    zp=2.50386e-9* ew**2 *1.e15 / c
    fluxval=nf*(ew**2 * 1.e-8 / c)
    new_mag = -2.5 * np.log10(fluxval / (zp * 1.e-23))
    return  nf,new_mag

def R_from_Teff_logg(Teff, logg,atm="thick"):
    from scipy import interpolate
    if atm=="thick":
        MGRID=pd.read_csv("new_MR_H.csv")
    elif atm=="thin":
        MGRID=pd.read_csv("CO_thinH_processed.csv")
    logT = np.log10(Teff)
    #logR=np.log10(R)
    #logR= interpolate.griddata((MGRID['logT'], MGRID['logg']), MGRID['logR'],(logT, logg))
    #R=10**(logR)
    R= interpolate.griddata((MGRID['logT'], MGRID['logg']), MGRID['R'],(logT, logg))
    return R
                        
def ex_d(x, a,b):
    return a * np.exp(b * (x**(-1/4)))
    
def bb(l,T,scale):
   
    f = 2*5.955e10 / l
    f /= l
    f /= l
    f /= l
    f /= l
    f /= (np.exp( 1.438e8 / (l*T) ) -1)
    f *= 1/scale
    return f

def line_info(wave,flux,err):
    flux=(flux/np.sum(flux))*np.size(flux)
    #bin spectrum for anchor points
    somma=np.sum(flux)
    numero=np.size(flux)
    scaling = (numero / somma)
    err = err * scaling
    flux = flux * scaling
    binsize=2
    xdata=[]
    ydata=[]
    edata=[]
    for i in range(0,(np.size(wave)-binsize),binsize):
        xdata.append(np.median(wave[i:i+binsize]))
        ydata.append(np.average(flux[i:i+binsize],weights=1/((err[i:i+binsize]))))
        edata.append(np.median(err[i:i+binsize]))
    wave_a=np.array(xdata)
    flux_a=np.array(ydata)
    err_a=np.array(edata)

    #define featurer lists
    feature_list=['DA.features', 'DB.features', 'DQ.features', 'DZ.features', 'WDMS.features','Pec.features','hDQ.features']

    #define anchor points for spline
    if np.max(wave_a)<8910:
        start_n=np.array([3850.,4230.,4600., 5300., 5700., 7350.,8100.,(np.max(wave_a)-10.)])
        start_pec=np.array([6000,6250.,6900., 7350.,8100])

    else:
        start_n=np.array([3850.,4230.,4600., 5300., 5700., 7350.,8100.,8900.])#,(np.max(wave_a)-10.)])
        start_pec=np.array([6000,6250.,6900., 7350.,8900])
    #define anchor points for WDMS line
    start_wdms=np.array([4600.,5100,5700, 6250.])#4230.

    # interpolate on predefined wavelength interval and range
    standard_wave=np.arange(3850,8300,1)#8300
    w_m=scipy.interpolate.interp1d(wave_a,flux_a)
    w_err=scipy.interpolate.interp1d(wave_a,err_a)
    flux=w_m(standard_wave)
    err=w_err(standard_wave)
    wave=standard_wave

    fluxes=[]
    errs=[]
    for xxx in start_n:
        interv=flux_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]#5
        int_err=err_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]
        try:
            fluxes.append(np.average(interv,weights=1/(int_err**2)))
        except:
            fluxes.append(np.average(interv))
        errs.append(np.mean(int_err))
    fluxes=np.array(fluxes)
    start_n=np.array(start_n)
    errs=np.array(errs)

    t0=[10000,1e-8]
    

    s = scipy.interpolate.InterpolatedUnivariateSpline(start_n,fluxes,w=1/errs)#,s=1)#InterpolatedUnivariateSpline
    y=s(wave)


    fluxes_wdms=[]
    errs_wdms=[]
    for xxx in start_wdms:
        interv=flux_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]
        int_err=err_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]
        try:
            fluxes_wdms.append(np.average(interv,weights=1/(int_err**2)))
        except:
            fluxes_wdms.append(np.average(interv))
        errs_wdms.append(np.mean(int_err))
    fluxes_wdms=np.array(fluxes_wdms)
    start_wdms=np.array(start_wdms)
    errs_wdms=np.array(errs_wdms)
  
    try:
        T_wdms,bla=scipy.optimize.curve_fit(bb,start_wdms,fluxes_wdms,[12000,1e-10],sigma=errs_wdms)
    except:
        T_wdms,bla=scipy.optimize.curve_fit(bb,start_wdms,fluxes_wdms,[6000,1e-10],sigma=errs_wdms)
    bbwdms=bb(wave,T_wdms[0],T_wdms[1])


    
    fluxes_pec=[]
    errs_pec=[]
    for xxx in start_pec:

        interv=flux_a[(wave_a>=xxx-15) & (wave_a<=xxx+15)]
        int_err=err_a[(wave_a>=xxx-15) & (wave_a<=xxx+15)]
        try:
            fluxes_pec.append(np.average(interv,weights=1/(int_err**2)))
        except:
            fluxes_pec.append(np.average(interv))
        errs_pec.append(np.mean(int_err))
    fluxes_pec=np.array(fluxes_pec)
    start_pec=np.array(start_pec)
    errs_pec=np.array(errs_pec)*10
    try:
        T_pec1,bla=scipy.optimize.curve_fit(bb,start_pec,fluxes_pec,[12000,1e-10],sigma=errs_pec)
    except:
        T_pec1,bla=scipy.optimize.curve_fit(bb,start_pec,fluxes_pec,[6000,1e-10],sigma=errs_pec)
    residuals = fluxes_pec- bb(start_pec, T_pec1[0],T_pec1[1])
    ss_res1 = np.sum(residuals**2)
   
    T_pec=T_pec1
    bbpec=bb(wave,T_pec[0],T_pec[1])
    bbpec[bbpec<1e-10]=1e-1
#===================Halpha core====================================
    #f_s=flux[np.logical_and(wave>6540,wave<6580)]
    #f_s=flux[(np.logical_and(wave>6540,wave<6555))|(np.logical_and(wave>6570,wave<6585))]
    #f_m=y[(np.logical_and(wave>6540,wave<6555))|(np.logical_and(wave>6570,wave<6585))]
    #ratio_line=f_s/f_m
    #minimum=np.min(ratio_line)
    minimum=np.mean(flux[(wave>6545)&(wave<6590)])
    core=np.max(flux[(wave>6555)&(wave<6570)])#np.mean(flux[np.logical_and(wave>6555,wave<6570)]/y[np.logical_and(wave>6555,wave<6570)])
    emission=core/minimum
    #print(emission,"BOOOM")
#==========================diagnostic plot========================================== 
    #plt.plot(wave_a,flux_a,c="0.7")
    #plt.plot(wave,y,c="b")
    #plt.plot(wave,bbwdms,c="m")
    #plt.plot(wave,bbpec,c="g")

    #plt.scatter(start_wdms,fluxes_wdms,c="m",zorder=3)
    #plt.scatter(start_pec,fluxes_pec,c="g",zorder=3)
    #plt.scatter(start_n,fluxes,c="r",zorder=3)

    #plt.scatter(np.mean(wave[(wave>6555)&(wave<6570)]),np.max(flux[(wave>6555)&(wave<6570)]),c="g")
    #plt.scatter(np.mean(wave[(wave>6550)&(wave<6580)]),np.mean(flux[(wave>6550)&(wave<6580)]),c="k")
    #plt.show()
#===================================================
    result={}
    for elem in feature_list:
        type=elem.rstrip('.features')
        features=[]
        start,end=np.loadtxt(elem,skiprows=1,delimiter='-', usecols=(0,1),unpack=True)
        for xxx in range(np.size(start)):
            if type=="WDMS":
                f_s=flux[np.logical_and(wave>start[xxx],wave<end[xxx])]
                f_m=bbwdms[np.logical_and(wave>start[xxx],wave<end[xxx])]
                ratio_line=np.average(f_s)/np.average(f_m)
                features.append(ratio_line)
            elif type=="Pec":
                f_s=flux[np.logical_and(wave>start[xxx],wave<end[xxx])]
                f_m=bbpec[np.logical_and(wave>start[xxx],wave<end[xxx])]
                ratio_line=np.average(f_s)/np.average(f_m)
                features.append(ratio_line)
            elif type=="hDQ":
                f_s=flux[np.logical_and(wave>start[xxx],wave<end[xxx])]
                f_m=bbpec[np.logical_and(wave>start[xxx],wave<end[xxx])]
                ratio_line=f_s/f_m#np.average(f_s)/np.average(f_m)
                features.extend(ratio_line)
            else:
                try:
                    f_s=flux[np.logical_and(wave>start[xxx],wave<end[xxx])]
                    f_m=y[np.logical_and(wave>start[xxx],wave<end[xxx])]
                    f_e=err[np.logical_and(wave>start[xxx],wave<end[xxx])]
                except:
                    f_s=flux[np.logical_and(wave>start,wave<end)]
                    f_m=y[np.logical_and(wave>start,wave<end)]
                    f_e=err[np.logical_and(wave>start,wave<end)]
                w=1/(f_e**2)
                w=w/(np.max(w))
                corr=(f_s-f_m)*abs(w-1)
                f_s=f_s-corr
                ratio_line=(f_s/f_m)
                features.extend(ratio_line)
        features=np.array(features)
        
#------------------------------------------------------------------------------------------------------------------
        result[type]=features
    emission2=np.mean(flux[np.logical_and(wave>6500,wave<6630)])/np.mean(y[np.logical_and(wave>6500,wave<6630)])
    all_lines=np.concatenate((result['DA'],result['DB'],result['DQ'],result['DZ'],result['WDMS'],result['Pec'],result['hDQ'],emission,emission2),axis=None)
    return all_lines
 
def WDclassify(wave,flux,err):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    with open("training_file_test", 'rb') as f:
        kf = pickle._load(f,fix_imports=True)
    err=err[(wave>=3650)&(wave<9800)]
    flux=flux[(wave>=3650)&(wave<9800)]
    wave=wave[(wave>=3650)&(wave<9800)]
        
    if (np.mean(flux)< 1e-34) or (np.size(flux)==0):
        print("No flux")
        print("WD classifier failed")
        p_class="UNKN"
        first=0.0
        return(p_class,first)
    else:
        err[err<=0.]=1
        somma=np.sum(flux)
        numero=np.size(flux)
        flux_temp=(flux/somma)*numero
        err_rel=err/flux
        err=err_rel*flux_temp
        flux=flux_temp
        try:
            labels= line_info(wave,flux,err)
            predictions = kf.predict(labels.reshape(1, -1))
            probs = kf.predict_proba(labels.reshape(1, -1))
                
            first= probs[0][kf.classes_==predictions[0]]
            #print(probs[0])
            print(kf.classes_)
            print(probs[0])
            if first >=0.5:
                p_class=predictions[0]
            else:
                second=sorted(probs[0])[-2]
                if second/first>0.6:
                    p_=predictions[0]+"/"+kf.classes_[probs[0]==second]
                    p_class=p_[0]
                else:
                    p_class=predictions[0]+":"
        except:
            print("WD classifier failed")
            p_class="UNKN"
            first=0.0
    return(p_class,first)


def fit_func(x,specn,lcrop,emu,wref,mode=0):
    """Requires: x - initial guess of T, g, and rv
       specn/lcrop - normalised spectrum / list of cropped lines to fit
       mode=0 is for finding bestfit, mode=1 for fitting & retriving specific model """
    
    tmp = tmp_func_rv(x[0], x[1],x[2], specn, lcrop, emu,wref,mode)
    if mode==0:
        return tmp[3] #this is the quantity that gets minimized   
    elif mode==1: return tmp[0], tmp[1], tmp[2]
    elif mode==2: return tmp[4]
    
def tmp_func_rv(_T, _g,_rv,_sn, _l, emu,wref,mode):
    c = 299792.458 # Speed of light in km/s=
    recovered=generate_modelDA(_T,(_g),emu)
    model=np.stack((wref,recovered),axis=-1)
    norm_model, m_cont_flux=norm_spectra(model)
    m_wave_n, m_flux_n, sn_w = norm_model[:,0], norm_model[:,1], _sn[:,0]
   
    lines_m, lines_s, sum_l_chi2 = [],[],0
    flux_s,err_s=[],[]
    chi_line=[]
    for i in range(len(_l)):
        vv=_rv
        if mode!=2:
            _l_c=_l*(vv+c)/c
            m_wave_n_c=m_wave_n*(vv+c)/c
        else:
            _l_c=_l
            m_wave_n_c=m_wave_n
        m_flux_n_i_c = interpolate.interp1d(m_wave_n_c,m_flux_n,kind='linear')(sn_w)
        m_flux_n_i=m_flux_n_i_c#*np.sum(_sn[:,1])/np.sum(m_flux_n_i_c)
        # Crop model and spec to line
        l_c0,l_c1 = _l_c[i,0],_l_c[i,1]
        l_m = m_flux_n_i.transpose()[(sn_w>=l_c0)&(sn_w<=l_c1)].transpose()
        l_s = _sn[(sn_w>=l_c0)&(sn_w<=l_c1)]
        lines_m.append(l_m)
        lines_s.append(l_s)
        flux_s.append(l_s[:,1])
        err_s.append(l_s[:,2])
        chi_line.append(np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2)/np.size(l_m))
    all_lines_m=np.concatenate((lines_m),axis=0)
    all_lines_s=np.concatenate((flux_s),axis=0)
    all_err_s=np.concatenate((err_s),axis=0)
    sum_l_chi2=((all_lines_s-all_lines_m)/all_err_s)**2
    chi_line=np.array(chi_line)
    chi_line_s=np.sort(chi_line)
    chi_line_m = chi_line_s[:-1]
    chi_sum=np.mean(chi_line)
    chi=np.array((all_lines_s-all_lines_m)/all_err_s)
    return lines_s, lines_m, model, sum_l_chi2,chi,chi_sum
