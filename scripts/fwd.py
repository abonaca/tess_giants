import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.table import Table
import astropy.units as u
from astropy.io import fits

import nfft

import subprocess
import os
import glob

def download_lightcurves():
    """"""
    tin = Table.read('../data/aguirre.txt', format='ascii')
    ids = tin['TIC']
    
    for id_ in ids[:]:
        for j in range(24):
            cmd = 'grep {:d} tess_bulk_download/tesscurl_sector_{:d}_lc.sh'.format(id_, j+1)
            cmd_list = cmd.split(' ')
            
            result = subprocess.run(cmd_list, stdout=subprocess.PIPE)
            
            if len(result.stdout):
                wd = os.getcwd()
                os.chdir('../data/tess_lightcurve')
                cmd = result.stdout.decode('utf-8')[:-1]
                cmd_list = cmd.split(' ')
                print(cmd_list)
                result = subprocess.run(cmd_list, stdout=subprocess.PIPE)
                os.chdir(wd)

def find_first_sector():
    """"""
    tin = Table.read('../data/aguirre.txt', format='ascii')
    ids = tin['TIC']
    
    tout = tin.copy(copy_data=True)
    
    fnames = []
    to_remove = []
    
    for e, id_ in enumerate(ids[:]):
        flist = glob.glob('../data/tess_lightcurve/*{:d}*'.format(id_))
        if len(flist)>0:
            fnames += [flist[0]]
        else:
            to_remove += [e]
    
    tout.remove_rows(to_remove)
    tout['fname'] = fnames
    
    tout.pprint()
    tout.write('../data/aguirre_1sec.fits', overwrite=True)

def plot_lightcurve_1sec():
    """"""
    
    tin = Table.read('../data/aguirre_1sec.fits')
    
    pp = PdfPages('../plots/aguirre_lightcurves.pdf')

    for i in range(len(tin)):
        t = Table(fits.getdata(tin['fname'][i], ignore_missing_end=True))
        
        plt.close()
        fig = plt.figure(figsize=(15,5))
        plt.plot(t['TIME'], t['PDCSAP_FLUX'], 'k-', lw=0.5)

        plt.xlabel('Time [days]')
        plt.ylabel('Flux')

        plt.tight_layout()
        pp.savefig(fig)
        
    pp.close()

def plot_nfft_1sec():
    """"""
    N = 600
    eta = 0.01
    N = int(N/eta)
    k_iday = (-int(N/2) + np.arange(N))*eta
    k = (k_iday*u.day**-1).to(u.uHz)
    
    tin = Table.read('../data/aguirre_1sec.fits')
    n = len(tin)
    
    pp = PdfPages('../plots/aguirre_nfft.pdf')

    for i in range(n):
        t = Table(fits.getdata(tin['fname'][i], ignore_missing_end=True))
        
        tm = t['TIME']
        fm = t['PDCSAP_FLUX']
        fm = fm - np.nanmean(fm)
        ind_finite = np.isfinite(fm)
        tm = tm[ind_finite]
        fm = fm[ind_finite]

        f_lc = nfft.nfft_adjoint(tm*eta, fm, len(k))
        amp = np.sqrt(f_lc.real**2 + f_lc.imag**2)
        
        plt.close()
        fig = plt.figure(figsize=(10,6))
        
        plt.plot(k, amp, '-', color='k', lw=0.8, label='TESS pipeline')
        plt.axvline(tin['numax'][i], label='True $\\nu_{max}$')

        #plt.legend(fontsize='small', loc=2)
        plt.xlim(10,300)
        plt.gca().set_xscale('log')

        plt.xlabel('Frequency [$\mu$Hz]')
        plt.ylabel('NFFT Amplitude')
        plt.tight_layout()
        pp.savefig(fig)
        
    pp.close()

def ln_profile_like_K_freqs(ts, ys, yivars, nu, deltanu, K=3):
    """Likelihood for a comb of K frequencies centered on nu and separated by deltanu (solves for the amplitudes)"""
    
    assert len(ts) == len(ys)
    
    halfK = (K - 1) // 2
    thisK = 2 * halfK + 1
    A = np.zeros((len(ts), 2 * thisK + 1))
    
    for k in range(thisK):
        f = nu - halfK * deltanu + k * deltanu
        A[:, 2 * k]     = np.cos(2. * np.pi * f * ts)
        A[:, 2 * k + 1] = np.sin(2. * np.pi * f * ts)
    
    A[:, -1] = 1.
    resid = ys - np.dot(A, np.linalg.solve(np.dot(A.T * yivars, A), np.dot(A.T * yivars, ys)))
    
    return -0.5 * np.sum(yivars * resid ** 2)

def ln_profile_like_K_freqs_unpack(ts, ys, yivars, nu, deltanu, K=3):
    """Likelihood for a comb of K frequencies centered on nu and separated by deltanu (solves for the amplitudes)"""
    
    assert len(ts) == len(ys)
    
    halfK = (K - 1) // 2
    thisK = 2 * halfK + 1
    A = np.zeros((len(ts), 2 * thisK + 1))
    
    for k in range(thisK):
        f = nu - halfK * deltanu + k * deltanu
        A[:, 2 * k]     = np.cos(2. * np.pi * f * ts)
        A[:, 2 * k + 1] = np.sin(2. * np.pi * f * ts)
    
    A[:, -1] = 1.
    Asolved = np.linalg.solve(np.dot(A.T * yivars, A), np.dot(A.T * yivars, ys))
    yp = np.dot(A, Asolved)
    resid = ys - yp
    chi2 = -0.5 * np.sum(yivars * resid ** 2)
    
    return (Asolved, chi2)


def dnu_numax(numax):
    """Relation for Kepler giants from Yu et al. (2018)"""
    
    dnu = 0.267 * numax**0.764
    
    return dnu


# diagnostics

def cross_dnu(K=3, ngrid=400):
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    
    i0 = 0
    t = Table(fits.getdata(tin['fname'][i0], ignore_missing_end=True))
    
    tm = t['TIME']
    fm = t['PDCSAP_FLUX']
    fm = fm - np.nanmean(fm)
    ind_finite = np.isfinite(fm)
    tm = tm[ind_finite]
    fm = fm[ind_finite]
    ivar = (t['PDCSAP_FLUX_ERR'][ind_finite])**-2
    
    numax = (59*u.uHz).to(u.day**-1).value
    
    dnu_min, dnu_max = 6.12, 6.14
    #freqs = np.linspace(numax_min, numax_max, ngrid)*u.uHz
    dfreqs = np.linspace(dnu_min, dnu_max, ngrid)*u.Hz

    #freqs_iday = freqs.to(u.day**-1).value
    dfreqs_iday = dfreqs.to(u.day**-1).value
    
    amps = np.empty((ngrid, 2*K+1))
    chi2 = np.empty(ngrid)
    
    for i in range(ngrid):
        amp, chi = ln_profile_like_K_freqs_unpack(tm, fm, ivar, numax, dfreqs_iday[i], K=K)
        amps[i] = amp
        chi2[i] = chi
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(12,8), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(dfreqs, chi2, 'k-')
    plt.ylabel('ln likelihood')
    
    plt.sca(ax[1])
    plt.plot(dfreqs, amps, '-')
    plt.ylabel('Amplitudes')
    plt.xlabel('$\Delta\\nu$ [$\mu$Hz]')
    
    plt.tight_layout()
    plt.savefig('../plots/cross_section_dnu_K.{:02d}.n.{:03d}.png'.format(K, ngrid))

def cross_numax(K=3, ngrid=400):
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    
    i0 = 0
    t = Table(fits.getdata(tin['fname'][i0], ignore_missing_end=True))
    
    tm = t['TIME']
    fm = t['PDCSAP_FLUX']
    fm = fm - np.nanmean(fm)
    ind_finite = np.isfinite(fm)
    tm = tm[ind_finite]
    fm = fm[ind_finite]
    ivar = (t['PDCSAP_FLUX_ERR'][ind_finite])**-2
    
    numax = (59*u.uHz).to(u.day**-1).value
    dnu = (6.125*u.uHz).to(u.day**-1).value
    
    numax_min, numax_max = 58, 60
    dnu_min, dnu_max = 6.12, 6.14
    freqs = np.linspace(numax_min, numax_max, ngrid)*u.uHz
    #dfreqs = np.linspace(dnu_min, dnu_max, ngrid)*u.Hz

    freqs_iday = freqs.to(u.day**-1).value
    #dfreqs_iday = dfreqs.to(u.day**-1).value
    
    amps = np.empty((ngrid, 2*K+1))
    chi2 = np.empty(ngrid)
    
    for i in range(ngrid):
        amp, chi = ln_profile_like_K_freqs_unpack(tm, fm, ivar, freqs_iday[i], dnu, K=K)
        amps[i] = amp
        chi2[i] = chi
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(12,8), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(freqs, chi2, 'k-')
    plt.ylabel('ln likelihood')
    
    plt.sca(ax[1])
    plt.plot(freqs, amps, '-')
    plt.ylabel('Amplitudes')
    plt.xlabel('$\\nu_{max}$ [$\mu$Hz]')
    
    plt.tight_layout()
    plt.savefig('../plots/cross_section_numax_K.{:02d}.n.{:03d}.png'.format(K, ngrid))

def nfft_highres():
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    n = len(tin)
    
    pp = PdfPages('../plots/aguirre_nfft_zoom.pdf')

    for i in range(n):
        t = Table(fits.getdata(tin['fname'][i], ignore_missing_end=True))
        numax = tin['numax'][i]
        dnu = tin['Delnu'][i]
        K = 4
        nu_min = numax - K*dnu
        nu_max = numax + K*dnu
        
        tm = t['TIME']
        fm = t['PDCSAP_FLUX']
        fm = fm - np.nanmean(fm)
        ind_finite = np.isfinite(fm)
        tm = tm[ind_finite]
        fm = fm[ind_finite]
        ivar = (t['PDCSAP_FLUX_ERR'][ind_finite])**-2
        
        N = 600
        eta = 0.001
        N = int(N/eta)
        k_iday = (-int(N/2) + np.arange(N))*eta
        k = (k_iday*u.day**-1).to(u.uHz)
        
        f_lc = nfft.nfft_adjoint(tm*eta, fm, len(k))
        amp = np.sqrt(f_lc.real**2 + f_lc.imag**2)
        
        plt.close()
        fig = plt.figure(figsize=(12,6))
        
        plt.plot(k, amp, 'k-')
        
        plt.axvline(numax, ls='-', color='tab:blue')
        plt.axvline(numax-dnu, ls=':', color='tab:blue')
        plt.axvline(numax+dnu, ls=':', color='tab:blue')
        
        plt.xlim(nu_min, nu_max)
        plt.xlabel('Frequency [$\mu$Hz]')
        plt.ylabel('NFFT Amplitude')
        
        plt.tight_layout()
        pp.savefig(fig)
        
    pp.close()



def time_frame():
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    t = Table(fits.getdata(tin['fname'][0], ignore_missing_end=True))
    print(t.colnames)
    tm = t['TIME']
    fm = -2.5*np.log10(t['PDCSAP_FLUX'])
    fm = fm - np.nanmean(fm)
    fe = -2.5*np.log10(t['PDCSAP_FLUX_ERR'])
    ind_finite = np.isfinite(fm)
    fm = fm[ind_finite]
    fe = fe[ind_finite]
    
    plt.close()
    plt.figure()
    
    plt.plot(tm[1:], tm[1:] - tm[:-1], 'k.')
    
    #plt.gca().set_yscale('log')
    plt.xlabel('time')
    plt.ylabel('d time')
    
    plt.tight_layout()
    
def find_nupeak_dnu():
    """"""

    tin = Table.read('../data/aguirre_1sec.fits')
    n = len(tin)
    
    for i0 in range(1):
        # load lightcurve
        t = Table(fits.getdata(tin['fname'][i0], ignore_missing_end=True))
        tm = t['TIME']
        fm = t['PDCSAP_FLUX']
        fm = fm - np.nanmean(fm)
        ind_finite = np.isfinite(fm)
        tm = tm[ind_finite]
        fm = fm[ind_finite]
        ivar = (np.ones_like(fm)*1e-8)**-1
        #ivar = (t['PDCSAP_FLUX_ERR'][ind_finite])**-2
        #print(ivar)
        
        # expected numax, delta nu
        numax = tin['numax'][i0]
        dnu = tin['Delnu'][i0]
        dnu_est = dnu_numax(numax)

        numax_err = np.sqrt(tin['Stnumax']**2 + tin['Synumax']**2)[i0]
        dnu_err = np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2)[i0]

        # reasonable range to search
        #nsigma = 3
        #numax_min = np.max(0, numax - nsigma*numax_err)
        #numax_max = numax + nsigma*numax_err
        
        nsigma = 1
        numax_min = max(0, numax - nsigma*dnu)
        numax_max = numax + nsigma*dnu
        
        nsigma = 5
        dnu_min = max(0, dnu - nsigma*dnu_err)
        dnu_max = dnu + nsigma*dnu_err
        
        numax_min, numax_max = 58, 60
        dnu_min, dnu_max = 6.12, 6.14

        # frequency grid
        ngrid = 100
        freqs = np.linspace(numax_min, numax_max, ngrid)*u.uHz
        dfreqs = np.linspace(dnu_min, dnu_max, ngrid)*u.Hz

        freqs_iday = freqs.to(u.day**-1).value
        dfreqs_iday = dfreqs.to(u.day**-1).value

        # 2D likelihood surface for a range of K
        Klist = [3,5,7]
        lls = np.zeros((len(Klist), len(dfreqs), len(freqs)))

        for i, f in enumerate(freqs_iday):
            for j, df in enumerate(dfreqs_iday):
                for k, K0 in enumerate(Klist):
                    lls[k, j, i] = ln_profile_like_K_freqs(tm, fm, ivar, f, df, K=K0)
        
        np.savez('../data/lhood_surface_guided_hres_{:09d}'.format(tin['TIC'][i0]), freqs=freqs, dfreqs=dfreqs, lls=lls)

def plot_lhood_2d():
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    n = len(tin)

    Klist = [3,5,7]
    pp = PdfPages('../plots/aguirre_lhood_surface_hres.pdf')
    
    #print((1/(27*u.day)).to(u.uHz))
    #print(np.percentile(np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2), [10,50]))
    
    for i0 in range(1):
        fin = np.load('../data/lhood_surface_guided_hres_{:09d}.npz'.format(tin['TIC'][i0]))
        freqs = fin['freqs']*u.uHz
        dfreqs = fin['dfreqs']*u.uHz
        lls = fin['lls']
        
        plt.close()
        fig, ax = plt.subplots(1,3,figsize=(15,6), sharex=True, sharey=True)

        for e in range(3):
            plt.sca(ax[e])
            numax_err = np.sqrt(tin['Stnumax']**2 + tin['Synumax']**2)[i0]
            dnu_err = np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2)[i0]
            
            pe = mpl.patches.Ellipse([tin['numax'][i0], tin['Delnu'][i0]], width=2*numax_err, height=2*dnu_err,
                                    facecolor='none', edgecolor='r', lw=2, zorder=10)
            plt.gca().add_patch(pe)

            plt.imshow(lls[e], origin='lower',
                    extent=[freqs[0].value, freqs[-1].value, dfreqs[0].value, dfreqs[-1].value],
                    aspect='auto', cmap='viridis')

            plt.xlim(freqs[0].value, freqs[-1].value)
            plt.ylim(dfreqs[0].value, dfreqs[-1].value)

            plt.title('K = {:d}'.format(Klist[e]), fontsize='medium')
            plt.xlabel('$\\nu$ [$\mu$Hz]')

        plt.sca(ax[0])
        plt.ylabel('$\Delta\\nu$ [$\mu$Hz]')

        plt.tight_layout()
        pp.savefig(fig)
    pp.close()
