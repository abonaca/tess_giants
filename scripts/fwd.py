import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.table import Table
import astropy.units as u
from astropy.io import fits

import nfft

from scipy.special import logsumexp
from scipy.signal import butter, filtfilt

import subprocess
import os
import glob
import time

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
        flist = glob.glob('../data/singletess_lightcurve/*{:d}*'.format(id_))
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
    
    # uncertainty tensor
    ayivar = A.T * yivars
    m = np.dot(ayivar, A)
    
    # vector
    Asolved = np.linalg.solve(m, np.dot(ayivar, ys))
    
    # scalar
    yp = np.dot(A, Asolved)
    resid = ys - yp
    chi2 = -0.5 * np.sum(yivars * resid ** 2)
    
    return (chi2, Asolved, m)

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
    cond = np.empty(ngrid)
    
    for i in range(ngrid):
        amp, cond_, resid_, chi = ln_profile_like_K_freqs_unpack(tm, fm, ivar, numax, dfreqs_iday[i], K=K)
        amps[i] = amp
        cond[i] = cond_
        chi2[i] = chi
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(12,8), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(dfreqs, chi2, 'k-')
    plt.ylabel('ln likelihood')
    
    plt.sca(ax[1])
    plt.plot(dfreqs, cond, 'k-')
    plt.ylabel('condition number')
    
    plt.sca(ax[2])
    plt.plot(dfreqs, amps, '-')
    plt.ylabel('Amplitudes')
    plt.xlabel('$\Delta\\nu$ [$\mu$Hz]')
    
    plt.tight_layout(h_pad=0)
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
        plt.text(0.05, 0.9, '{:d}'.format(tin['TIC'][i]), transform=plt.gca().transAxes)
        
        plt.xlim(nu_min, nu_max)
        plt.xlabel('Frequency [$\mu$Hz]')
        plt.ylabel('NFFT Amplitude')
        
        plt.tight_layout()
        pp.savefig(fig)
        
    pp.close()

def jump_residuals(K=3, ngrid=400):
    tin = Table.read('../data/aguirre_1sec.fits')
    
    i0 = 0
    t = Table(fits.getdata(tin['fname'][i0], ignore_missing_end=True))
    
    #print(t['PDCSAP_FLUX'], t['PDCSAP_FLUX_ERR'])
    
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
    cond = np.empty(ngrid)
    
    print(np.min(ivar), np.max(ivar), np.median(ivar))
    
#def br():
    for i in range(ngrid):
        amp, cond_, resid_, chi = ln_profile_like_K_freqs_unpack(tm, fm, ivar, numax, dfreqs_iday[i], K=K)
        amps[i] = amp
        cond[i] = cond_
        chi2[i] = chi
    
    # find the jump
    jump = chi2[1:] - chi2[:-1]
    ind_jump = jump>1000
    #ind_jump = np.concatenate([ind_jump, False])
    indices = np.arange(ngrid, dtype=int)
    
    ijump = indices[:-1][ind_jump][0] + 1
    ipre = ijump - 1
    
    amp_pre, cond_pre, resid_pre, chi_pre = ln_profile_like_K_freqs_unpack(tm, fm, ivar, numax, dfreqs_iday[ipre], K=K)
    amp_jump, cond_jump, resid_jump, chi_jump = ln_profile_like_K_freqs_unpack(tm, fm, ivar, numax, dfreqs_iday[ijump], K=K)
    
    print(chi_pre, chi_jump)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(15,10))
    
    plt.sca(ax[0])
    plt.plot(tm, ivar*resid_pre**2, 'k-')
    plt.plot(tm, ivar*resid_jump**2, 'r-')
    
    plt.sca(ax[1])
    plt.plot(tm, ivar*(resid_pre**2 - resid_jump**2), 'k-')
    plt.plot(tm, np.cumsum(ivar*(resid_pre**2 - resid_jump**2)), 'k-', alpha=0.5)
    
    plt.tight_layout()


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
        dfreqs = np.linspace(dnu_min, dnu_max, ngrid)*u.uHz

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
            
            pe = mpl.patches.Ellipse([tin['numax'][i0], tin['Delnu'][i0]], width=2*numax_err, height=2*dnu_err, facecolor='none', edgecolor='r', lw=2, zorder=10)
            plt.gca().add_patch(pe)

            plt.imshow(lls[e], origin='lower', extent=[freqs[0].value, freqs[-1].value, dfreqs[0].value, dfreqs[-1].value], aspect='auto', cmap='viridis')

            plt.xlim(freqs[0].value, freqs[-1].value)
            plt.ylim(dfreqs[0].value, dfreqs[-1].value)

            plt.title('K = {:d}'.format(Klist[e]), fontsize='medium')
            plt.xlabel('$\\nu$ [$\mu$Hz]')

        plt.sca(ax[0])
        plt.ylabel('$\Delta\\nu$ [$\mu$Hz]')

        plt.tight_layout()
        pp.savefig(fig)
    pp.close()
    

## multi-sector ##

def full_lightcurve(tic, verbose=True):
    """"""
    flist = glob.glob('../data/tess_lightcurve/*{:d}*'.format(tic))
    
    tm_full = np.empty(0)
    fm_full = np.empty(0)
    flux_full = np.empty(0)
    ivar_full = np.empty(0)
    sector_full = np.empty(0)
    
    for f in flist:
        t = Table(fits.getdata(f, ignore_missing_end=True))
        sector = int(f.split('-')[1][1:])
        
        tm = t['TIME']
        flux = t['PDCSAP_FLUX']
        fm = flux - np.nanmean(flux)
        ind_finite = np.isfinite(flux)
        tm = tm[ind_finite]
        fm = fm[ind_finite]
        flux = flux[ind_finite]
        ivar = (t['PDCSAP_FLUX_ERR'][ind_finite])**-2
        sector = sector * np.ones(np.sum(ind_finite), dtype=int)
        
        tm_full = np.concatenate((tm_full, tm))
        fm_full = np.concatenate((fm_full, fm))
        flux_full = np.concatenate((flux_full, flux))
        ivar_full = np.concatenate((ivar_full, ivar))
        sector_full = np.concatenate((sector_full, sector))
    
    tout = Table([tm_full, fm_full, flux_full, ivar_full, sector_full], names=('time', 'flux', 'flux_raw', 'ivar', 'sector'), dtype=('float', 'float', 'float', 'float', 'int'))
    if verbose: tout.pprint()
    tout.write('../data/full_lightcurve/{:016d}.fits'.format(tic), overwrite=True)

def aguirre_full_lightcurves(verbose=True):
    """"""
    tin = Table.read('../data/aguirre.txt', format='ascii')
    tic = tin['TIC']
    
    for tic in tin['TIC']:
        if verbose: print(tic)
        full_lightcurve(tic, verbose=verbose)

def find_nupeak_dnu_full(single=False, mock=False, hres=False, fz=1):
    """"""

    tin = Table.read('../data/aguirre.txt', format='ascii')
    n = len(tin)
    
    if mock:
        lclabel = 'mock'
    else:
        lclabel = 'tess'
    
    for i0 in range(3,4):
        # load lightcurve
        t = Table(fits.getdata('../data/{:s}_lightcurve/{:016d}.fits'.format(lclabel, tin['TIC'][i0])))
        if single:
            ind = t['sector']==np.unique(t['sector'])[0]
            t = t[ind]
            label = 'single'
        else:
            label = 'full'
        
        # expected numax, delta nu
        numax = tin['numax'][i0]
        dnu = tin['Delnu'][i0]
        dnu_est = dnu_numax(numax)

        numax_err = np.sqrt(tin['Stnumax']**2 + tin['Synumax']**2)[i0]
        dnu_err = np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2)[i0]

        # reasonable range to search
        nsigma = fz*3
        numax_min = max(0, numax - nsigma*dnu)
        numax_max = numax + nsigma*dnu
        
        nsigma = fz*5
        dnu_min = max(0, dnu - nsigma*dnu_err)
        dnu_max = dnu + nsigma*dnu_err
        
        # frequency grid
        if hres:
            ngrid = 200
            reslabel = 'hres'
        else:
            ngrid = 100
            reslabel = 'lres'
        freqs = np.linspace(numax_min, numax_max, ngrid)*u.uHz
        dfreqs = np.linspace(dnu_min, dnu_max, ngrid)*u.uHz

        freqs_iday = freqs.to(u.day**-1).value
        dfreqs_iday = dfreqs.to(u.day**-1).value

        print(freqs[1]-freqs[0], dfreqs[1]-dfreqs[0])
        print(freqs_iday[1]-freqs_iday[0], dfreqs_iday[1]-dfreqs_iday[0], 1/(np.max(t['time'])-np.min(t['time'])))

        # 2D likelihood surface for a range of K
        Klist = [3,]
        lls = np.zeros((len(Klist), len(dfreqs), len(freqs)))

        for i, f in enumerate(freqs_iday):
            for j, df in enumerate(dfreqs_iday):
                for k, K0 in enumerate(Klist):
                    lls[k, j, i] = ln_profile_like_K_freqs(t['time'], t['flux'], t['ivar'], f, df, K=K0)
        
        np.savez('../data/lhood_surface_{:s}_{:09d}_{:s}_{:s}_K{:d}_fz{:.1f}'.format(label, tin['TIC'][i0], lclabel, reslabel, len(Klist), fz), freqs=freqs, dfreqs=dfreqs, lls=lls)

def run_comparison():
    """"""
    for single in [True, False]:
        for mock in [True, False]:
            print(single, mock)
            find_nupeak_dnu_full(single=single, mock=mock, hres=True, fz=1)


def plot_lhood_comparison(i0=0, label='single'):
    """"""
    tin = Table.read('../data/aguirre.txt', format='ascii')
    n = len(tin)

    Klist = [3,5,7]
    
    plt.close()
    fig, ax = plt.subplots(2,3,figsize=(17,10), sharex=True, sharey=True)

    labels = ['single', 'full']
    fnames = ['../data/lhood_surface_{:s}_{:09d}_full.npz'.format(label, tin['TIC'][i0]), '../data/lhood_surface_{:s}_{:09d}_mock.npz'.format(label, tin['TIC'][i0])]
    row_labels = ['TIC {:d}'.format(tin['TIC'][i0]), 'mock {:d}'.format(tin['TIC'][i0])]
    
    for er, fname in enumerate(fnames):
        fin = np.load(fname)
        freqs = fin['freqs']*u.uHz
        dfreqs = fin['dfreqs']*u.uHz
        lls = fin['lls']
        
        for e in range(3):
            plt.sca(ax[er][e])
            numax_err = np.sqrt(tin['Stnumax']**2 + tin['Synumax']**2)[i0]
            dnu_err = np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2)[i0]
            
            pe = mpl.patches.Ellipse([tin['numax'][i0], tin['Delnu'][i0]], width=2*numax_err, height=2*dnu_err, facecolor='none', edgecolor='r', lw=2, zorder=10)
            plt.gca().add_patch(pe)

            vmin, vmax = np.percentile(lls[e], [0.1,99.9])
            
            im = plt.imshow(lls[e], origin='lower', extent=[freqs[0].value, freqs[-1].value, dfreqs[0].value, dfreqs[-1].value], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

            plt.xlim(freqs[0].value, freqs[-1].value)
            plt.ylim(dfreqs[0].value, dfreqs[-1].value)
            
            plt.colorbar()

        plt.sca(ax[er][0])
        plt.ylabel('$\Delta\\nu$ [$\mu$Hz]')
        plt.text(0.1, 0.9, row_labels[er], fontsize='small', color='w', zorder=10, transform=plt.gca().transAxes)

    for i in range(3):
        plt.sca(ax[0][i])
        plt.title('K = {:d}'.format(Klist[i]), fontsize='medium')
        
        plt.sca(ax[1][i])
        plt.xlabel('$\\nu$ [$\mu$Hz]')

    plt.tight_layout()
    plt.savefig('../plots/lhood2d_comparison_{:s}_{:09d}.png'.format(label, tin['TIC'][i0]))

def multi_comparison(i0=20, fz=1, voff=0, scale=False):
    """"""
    
    tin = Table.read('../data/aguirre.txt', format='ascii')
    tic = tin['TIC'][i0]
    numax_err = np.sqrt(tin['Stnumax']**2 + tin['Synumax']**2)[i0]
    dnu_err = np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2)[i0]
    
    d = []
    sectors = ['single', 'full']
    sources = ['tess', 'mock']
    
    for sector in sectors:
        for source in sources:
            fname = '../data/lhood_surface_{:s}_{:09d}_{:s}_hres_K1_fz{:.1f}.npz'.format(sector, tic, source, fz)
            fin = np.load(fname)
            d += [fin]

    plt.close()
    fig, ax = plt.subplots(2,2, figsize=(14,12))
    
    for i in range(4):
        freqs = d[i]['freqs']*u.uHz
        dfreqs = d[i]['dfreqs']*u.uHz
        lls = d[i]['lls'][0]
        
        irow = int(i/2)
        icol = i%2
        plt.sca(ax[irow][icol])
        #print(i, irow, icol)
        
        if voff==0:
            vmin, vmax = np.percentile(lls, [0.1, 99.9])
        else:
            vmax = np.max(lls)
            vmin = vmax - voff
            
        #vmin, vmax = np.percentile(lls, [99.89999,99.9])
        im = plt.imshow(lls, origin='lower', extent=[freqs[0].value, freqs[-1].value, dfreqs[0].value, dfreqs[-1].value], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        
        pe = mpl.patches.Ellipse([tin['numax'][i0], tin['Delnu'][i0]], width=2*numax_err, height=2*dnu_err, facecolor='none', edgecolor='darkorange', lw=2, zorder=10)
        plt.gca().add_patch(pe)
        
        numax_ = np.linspace(freqs[0].value, freqs[-1].value, 100)
        dnu_ = dnu_numax(numax_)
        
        plt.plot(numax_, dnu_, 'r-', lw=2, zorder=10)
        # 0.85, 1.2
        for p in [0.95, 1.07]:
            plt.plot(numax_, p*dnu_, '--', color='r', lw=2, zorder=10)
        
        plt.xlim(freqs[0].value, freqs[-1].value)
        plt.ylim(dfreqs[0].value, dfreqs[-1].value)
        plt.text(0.05, 0.95, '{:s}'.format(sectors[irow]), fontsize='small', transform=plt.gca().transAxes, color='0.9', va='top')
        
        plt.colorbar()
        if scale:
            plt.gca().set_aspect('equal')
    
    for i in range(2):
        plt.sca(ax[0][i])
        plt.title('{:s} {:d}'.format(sources[i], tic), fontsize='medium')
        
        plt.sca(ax[1][i])
        plt.xlabel('$\\nu$ [$\mu$Hz]')
        
        plt.sca(ax[i][0])
        plt.ylabel('$\Delta_{\\nu}$ [$\mu$Hz]')
    
    plt.tight_layout()
    plt.savefig('../plots/mock_comparison_{:09d}_fz{:.1f}_voff{:04d}_scale{:d}.png'.format(tic, fz, voff, scale))

def nfft_comparison():
    """"""
    
    N = 600
    eta = 0.01
    N = int(N/eta)
    k_iday = (-int(N/2) + np.arange(N))*eta
    k = (k_iday*u.day**-1).to(u.uHz)
    
    tin = Table.read('../data/aguirre.txt', format='ascii')
    n = len(tin)
    
    pp = PdfPages('../plots/aguirre_nfft_comparison.pdf')

    for i in range(n):
        # load lightcurve
        t = Table(fits.getdata('../data/full_lightcurve/{:016d}.fits'.format(tin['TIC'][i])))
        
        if len(t)>0:
            ind = t['sector']==np.unique(t['sector'])[0]
            t1 = t[ind]
            
            tables = [t1, t]
            colors = ['k', 'k']
            labels = ['1 sector', 'All ({:d}) sectors'.format(np.size(np.unique(t['sector'])))]
            
            plt.close()
            fig, ax = plt.subplots(2,1,figsize=(12,10), sharex=True)
            
            for j in range(2):
                f_lc = nfft.nfft_adjoint(tables[j]['time']*eta, tables[j]['flux'], len(k))
                amp = np.sqrt(f_lc.real**2 + f_lc.imag**2)
                
                plt.sca(ax[j])
                plt.plot(k, amp, '-', color=colors[j], lw=0.8, label=labels[j])
            
                plt.axvline(tin['numax'][i], label='Literature $\\nu_{max}$')
                plt.ylabel('NFFT Amplitude')
                plt.text(0.06, 0.9, labels[j], transform=plt.gca().transAxes)
            
                #plt.legend(fontsize='small', loc=2)
                plt.xlim(10,300)
                plt.gca().set_xscale('log')

            plt.xlabel('Frequency [$\mu$Hz]')
            plt.tight_layout()
            pp.savefig(fig)
        
    pp.close()

def lightcurve_comparison():
    """"""
    tin = Table.read('../data/aguirre.txt', format='ascii')
    n = len(tin)
    label = ['full', 'mock']
    
    pp = PdfPages('../plots/mock_lightcurve_comparison.pdf')

    for i in range(n):
        plt.close()
        fig, ax = plt.subplots(2,1,figsize=(15,10), sharex=True)
        row_labels = ['TIC {:d}'.format(tin['TIC'][i]), 'mock {:d}'.format(tin['TIC'][i])]
        
        for j in range(2):
            t = Table(fits.getdata('../data/{:s}_lightcurve/{:016d}.fits'.format(label[j], tin['TIC'][i])))
            if len(t)>0:
                plt.sca(ax[j])
                plt.plot(t['time'], t['flux'], 'k-', lw=0.5, rasterized=True)
                plt.ylabel('Flux')
                plt.text(0.03, 0.9, row_labels[j], fontsize='small', zorder=10, transform=plt.gca().transAxes)

        plt.xlabel('Time [days]')
        plt.tight_layout()
        pp.savefig(fig)
        
    pp.close()


# mock data

def create_mock_lightcurves():
    """"""
    
    tin = Table.read('../data/aguirre.txt', format='ascii')
    n = len(tin)
    
    np.random.seed(1903)
    
    for i in range(n):
        t = Table(fits.getdata('../data/full_lightcurve/{:016d}.fits'.format(tin['TIC'][i])))
        N = len(t)
        fmock = np.random.randn(N) / np.sqrt(t['ivar'])
        
        tout = t.copy(copy_data=True)
        tout['flux'] = fmock
        tout.write('../data/mock_lightcurve/{:016d}.fits'.format(tin['TIC'][i]), overwrite=True)
        
        #plt.close()
        #plt.plot(tout['time'], tout['flux'], 'k-')
        #plt.tight_layout()

def dnu_profile(i0=20, fz=1):
    """"""
    
    tin = Table.read('../data/aguirre.txt', format='ascii')
    tic = tin['TIC'][i0]
    numax_err = np.sqrt(tin['Stnumax']**2 + tin['Synumax']**2)[i0]
    dnu_err = np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2)[i0]
    
    d = []
    sectors = ['single', 'full']
    sources = ['tess', 'mock']

    for sector in sectors:
        for source in sources:
            fname = '../data/lhood_surface_{:s}_{:09d}_{:s}_hres_K1_fz{:.1f}.npz'.format(sector, tic, source, fz)
            fin = np.load(fname)
            d += [fin]
    
    numax_0 = tin['numax'][i0] - 2*numax_err
    numax_1 = tin['numax'][i0] + 2*numax_err
    
    color = ['k', '0.7']
    lw = [2, 1]
    
    plt.close()
    plt.figure(figsize=(8,8))
    
    for i in range(4):
        irow = int(i/2)
        icol = i%2
        
        freqs = d[i]['freqs']
        dfreqs = d[i]['dfreqs']*u.uHz
        lls = d[i]['lls'][0]
        
        numax_0, numax_1 = 71, 75
        
        x0 = np.argmin(np.abs(freqs - numax_0))
        x1 = np.argmin(np.abs(freqs - numax_1))
        #x0 = 110
        #x1 = 112
        print(x0, x1)
        
        # lls[dnu, numax]
        dnu_ = np.mean(lls[:,x0:x1], axis=1)
        dnu_ = dnu_ - np.min(dnu_)
        plt.plot(dfreqs[::], dnu_, '-', color=color[icol], lw=lw[irow], label='{:s} {:s}'.format(sources[icol], sectors[irow]))
        
        #plt.plot(dfreqs, lls[:,110], '-')
        #for j in range(0,200,40):
            #print(freqs[j])
            #plt.plot(dfreqs, lls[:,j], '-')
        
        #plt.plot(freqs, lls[100], '-')
    
    plt.xlabel('$\Delta\\nu$ [$\mu$Hz]')
    plt.ylabel('<$\Delta$ likelihood>')
    plt.legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig('../plots/dnu_profile_{:09d}.png'.format(tic))


# fit numax / marginalize nupeak

def find_numax_dnu(mg='mean', filter=False, zoom=False, n_dnu=50):
    """"""
    
    tin = Table.read('../data/aguirre_1sec.fits')
    n = len(tin)
    
    for i0 in range(13,14):
        # load lightcurve
        t = Table(fits.getdata(tin['fname'][i0], ignore_missing_end=True))
        tm = t['TIME']
        fm = t['PDCSAP_FLUX']
        fm = (fm - np.nanmean(fm))/np.nanmean(fm)
        ind_finite = np.isfinite(fm)
        tm = tm[ind_finite]
        fm = fm[ind_finite]
        ivar = (np.ones_like(fm)*1e-8)**-1
        
        if filter:
            fs = 0.5/(tm[1]-tm[0])
            flow = (40*u.uHz).to(u.day**-1).value/fs
            fhigh = (110*u.uHz).to(u.day**-1).value/fs
            order = 9
            
            b, a = butter(order, flow, btype='high', analog=False)
            fm_band = filtfilt(b, a, fm)

            b, a = butter(order, fhigh, btype='low', analog=False)
            fm_band = filtfilt(b, a, fm_band)
            
            fm = fm_band
        
        # expected numax, delta nu
        numax = tin['numax'][i0]
        dnu = tin['Delnu'][i0]
        dnu_est = dnu_numax(numax)

        numax_err = np.sqrt(tin['Stnumax']**2 + tin['Synumax']**2)[i0]
        dnu_err = np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2)[i0]
        print(numax, dnu, dnu_est, numax_err, dnu_err)

        nsigma = 4
        numax_min = max(0, numax - nsigma*dnu)
        numax_max = numax + nsigma*dnu
        
        nsigma = 20
        dnu_min = max(0, dnu - nsigma*dnu_err)
        dnu_max = dnu + nsigma*dnu_err
        
        if zoom:
            numax_cen = 71.5
            numax_width = 0.5
            numax_min, numax_max = numax_cen - numax_width, numax_cen + numax_width
            dnu_cen = 7.2
            dnu_width = 0.5
            dnu_min, dnu_max = dnu_cen - dnu_width, dnu_cen + dnu_width
            

        # frequency grid
        n_numax = n_dnu
        freqs = np.linspace(numax_min, numax_max, n_numax)*u.uHz
        dfreqs = np.linspace(dnu_min, dnu_max, n_dnu)*u.uHz
        print('dnu {:.3f} numax {:.3f}'.format(dfreqs[1]-dfreqs[0], freqs[1]-freqs[0]))

        # nu_peak grid
        T = (np.max(tm)-np.min(tm))*u.day
        n_nupeak = 50
        n_nupeak = 1*int(np.ceil((2*T*dnu_max*u.uHz).decompose()))
        #print(n_nupeak)
        
        #print(freqs[1]-freqs[0])
        #print(dfreqs[1]-dfreqs[0])
        #print((((np.max(tm)-np.min(tm))*u.day)**-1).to(u.uHz))
        #print(dnu_min/n_nupeak, dnu_max/n_nupeak)

        freqs_iday = freqs.to(u.day**-1).value
        dfreqs_iday = dfreqs.to(u.day**-1).value

        # 2D likelihood surface for a range of K
        Klist = [5,]
        lls = np.zeros((len(Klist), len(dfreqs), len(freqs)))
        ll_nupeak = np.zeros((len(Klist), len(dfreqs), len(freqs), n_nupeak))

        for i, f in enumerate(freqs_iday[:]):
            for j, df in enumerate(dfreqs_iday[:]):
                for k, K0 in enumerate(Klist):
                    if mg=='none':
                        lls[k, j, i] = ln_profile_like_K_freqs(tm, fm, ivar, f, df, K=K0)
                    else:
                        # grid in nupeak
                        ddf = 2*df/n_nupeak
                        freqs_peak_iday = np.arange(f-df+0.5*ddf, f+df, ddf)
                        lgrid = np.zeros(n_nupeak)
                        
                        # evaluate likelihoods on the nupeak grid
                        for l, fp in enumerate(freqs_peak_iday):
                            lgrid[l] = ln_profile_like_K_freqs(tm, fm, ivar, fp, df, K=K0)
                        ll_nupeak[k, j, i] = lgrid
                        
                        # marginalize over the nupeak grid
                        if mg=='max':
                            lls[k, j, i] = np.max(lgrid)
                        else:
                            mg = 'mean'
                            lls[k, j, i] = logsumexp(lgrid) - np.log(np.size(lgrid))
                    
        np.savez('../data/lhood_surface_{:s}_{:d}_{:d}_zoom{:d}_numax_dnu_{:09d}'.format(mg, filter, n_dnu, zoom, tin['TIC'][i0]), freqs=freqs, dfreqs=dfreqs, lls=lls, ll_nupeak=ll_nupeak)

def plot_numax_dnu_lhood(k=0, mg='mean', filter=False, n_dnu=20, zoom=False):
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    n = len(tin)

    Klist = [5,]
    #pp = PdfPages('../plots/aguirre_lhood_surface_hres.pdf')
    
    #print((1/(27*u.day)).to(u.uHz))
    #print(np.percentile(np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2), [10,50]))
    
    for i0 in range(13,14):
        fin = np.load('../data/lhood_surface_{:s}_{:d}_{:d}_zoom{:d}_numax_dnu_{:09d}.npz'.format(mg, filter, n_dnu, zoom, tin['TIC'][i0]))
        freqs = fin['freqs']*u.uHz
        dfreqs = fin['dfreqs']*u.uHz
        lls = fin['lls']
        
        plt.close()
        fig, ax = plt.subplots(1,1,figsize=(7,6), sharex=True, sharey=True)

        for e in range(k, k+1):
            plt.sca(ax)
            numax_err = np.sqrt(tin['Stnumax']**2 + tin['Synumax']**2)[i0]
            dnu_err = np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2)[i0]
            
            pe = mpl.patches.Ellipse([tin['numax'][i0], tin['Delnu'][i0]], width=2*numax_err, height=2*dnu_err, facecolor='none', edgecolor='r', lw=2, zorder=10)
            plt.gca().add_patch(pe)

            im = plt.imshow(lls[e], origin='lower', extent=[freqs[0].value, freqs[-1].value, dfreqs[0].value, dfreqs[-1].value], aspect='auto', cmap='viridis', vmin=np.max(lls[e])-1000)

            plt.xlim(freqs[0].value, freqs[-1].value)
            plt.ylim(dfreqs[0].value, dfreqs[-1].value)

            plt.title('K = {:d}'.format(Klist[e]), fontsize='medium')
            plt.xlabel('$\\nu_{max}$ [$\mu$Hz]')
            
            plt.colorbar(label='ln Likelihood')

        if zoom:
            t = Table(fits.getdata(tin['fname'][i0], ignore_missing_end=True))
            tm = t['TIME']
            fm = t['PDCSAP_FLUX']
            fm = (fm - np.nanmean(fm))/np.nanmean(fm)
            ind_finite = np.isfinite(fm)
            tm = tm[ind_finite]
            fm = fm[ind_finite]
            
            freq_res = (((np.max(tm)-np.min(tm))*u.day)**-1).to(u.uHz)
            freq_spacing = (((tm[1]-tm[0])*u.day)**-1).to(u.uHz)
            print(freq_res, freq_spacing)
            
            x = np.array([-0.5,0.5])*freq_res + freqs[int(n_dnu*0.5)]
            y = np.ones(2)*dfreqs[int(n_dnu*0.9)]
            plt.plot(x, y, '-', color='orangered')
            
            x = np.array([-0.5,0.5])*freq_res/Klist[0] + freqs[int(n_dnu*0.5)]
            y = np.ones(2)*dfreqs[int(n_dnu*0.85)]
            plt.plot(x, y, '-', color='orangered')


        plt.sca(ax)
        plt.ylabel('$\Delta\\nu$ [$\mu$Hz]')

        plt.tight_layout()
        plt.savefig('../plots/marginalized_{:s}_{:d}_{:d}_zoom{:d}_numax_dnu_{:09d}'.format(mg, filter, n_dnu, zoom, tin['TIC'][i0]))
        #pp.savefig(fig)
    #pp.close()

def plot_nupeak_grid(k=0, i0=13, mg='mean', filter=False, n_dnu=20, zoom=False):
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    n = len(tin)

    Klist = [5,]
    
    fin = np.load('../data/lhood_surface_{:s}_{:d}_{:d}_zoom{:d}_numax_dnu_{:09d}.npz'.format(mg, filter, n_dnu, zoom, tin['TIC'][i0]))
    freqs = fin['freqs']*u.uHz
    dfreqs = fin['dfreqs']*u.uHz
    lls = fin['lls']
    ll_nupeak = fin['ll_nupeak']
    freqs = fin['freqs']
    dfreqs = fin['dfreqs']
    n_nupeak = np.shape(ll_nupeak)[-1]
    
    fin_nomg = np.load('../data/lhood_surface_{:s}_{:d}_{:d}_zoom{:d}_numax_dnu_{:09d}.npz'.format('none', filter, n_dnu, zoom, tin['TIC'][i0]))
    
    i = 4
    
    plt.close()
    plt.figure(figsize=(10,7))
    
    for j in range(n_dnu):
        df = dfreqs[j]
        f = freqs[i]
        ddf = 2*df/n_nupeak
        freqs_peak = np.arange(f-df+0.5*ddf, f+df, ddf)*u.uHz
        
        #color=mpl.cm.viridis(j/n_dnu)
        plt.plot(freqs_peak, ll_nupeak[k, j, i], '-', label='{:.2f}'.format(dfreqs[j]))

        #plt.plot(fin_nomg['freqs'], fin_nomg['lls'][k,j], '--', label='{:.2f}'.format(fin_nomg['dfreqs'][j]))
        
        #print(logsumexp(ll_nupeak[k,j,i]) - lls[k,j,i] - np.log(n_nupeak))
    
    #plt.xlim(71, 72)
    plt.legend(fontsize='small', loc=1, ncol=3)
    
    plt.tight_layout()


# search space

def search_grid():
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    N = len(tin)
    
    numax = np.linspace(0,250,100)
    dnu = dnu_numax(numax)
    
    xerr = 4*tin['Delnu']
    yerr = 20*np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2)
    
    xerr = np.ones(N) * 15
    yerr = np.ones(N) * 3
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    plt.plot(numax, dnu, 'r-', lw=4, alpha=0.4)
    plt.plot(numax, dnu+2.5, 'r-')
    plt.plot(numax, dnu-2.5, 'r-')
    plt.plot(numax, dnu*0.9, 'r--')
    plt.plot(numax, dnu*1.1, 'r--')
    plt.plot(numax, dnu*0.8, 'r:')
    plt.plot(numax, dnu*1.2, 'r:')
    
    plt.plot(tin['numax'], tin['Delnu'], 'ko')
    plt.errorbar(tin['numax'], tin['Delnu'], yerr=yerr, xerr=xerr, fmt='none', color='k', alpha=0.4)
    
    plt.xlim(0, 220)
    plt.ylim(0, 20)
    plt.xlabel('$\\nu_{max}$ [$\mu$Hz]')
    plt.ylabel('$\Delta\\nu$ [$\mu$Hz]')
    #plt.gca().set_aspect('equal')
    
    plt.tight_layout()

def bandpass_filter(fm, fmin, fmax, fs, order=9):
    """"""
    #fs = 0.5/(tm[1]-tm[0])
    flow = (fmin*u.uHz).to(u.day**-1).value/fs
    fhigh = (fmax*u.uHz).to(u.day**-1).value/fs
    #order = 9
    
    b, a = butter(order, flow, btype='high', analog=False)
    fm_band = filtfilt(b, a, fm)

    b, a = butter(order, fhigh, btype='low', analog=False)
    fm_band = filtfilt(b, a, fm_band)
    
    return fm_band

def search(i0=0, K0=5, mg='mean', filter=False):
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    
    # load lightcurve
    t = Table(fits.getdata(tin['fname'][i0], ignore_missing_end=True))
    tm = t['TIME']
    fs = 0.5/(tm[1]-tm[0])
    fm = t['PDCSAP_FLUX']
    fm = (fm - np.nanmean(fm))/np.nanmean(fm)
    
    ind_finite = np.isfinite(fm)
    tm = tm[ind_finite]
    fm = fm[ind_finite]
    ivar = (np.ones_like(fm)*1e-8)**-1
    print(np.size(ivar))
    
    # expected numax, delta nu
    numax = tin['numax'][i0]
    dnu = tin['Delnu'][i0]
    
    # lay down the search grid
    w_numax = 15
    w_dnu = 3
    res_numax = 1
    res_dnu = 0.1
    n_numax = int(2*w_numax/res_numax)
    n_dnu = int(2*w_dnu/res_dnu)
    
    freqs = np.linspace(numax-w_numax, numax+w_numax, n_numax)*u.uHz
    dfreqs = np.linspace(dnu-w_dnu, dnu+w_dnu, n_dnu)*u.uHz
    print('dnu {:.3f} numax {:.3f}'.format(dfreqs[1]-dfreqs[0], freqs[1]-freqs[0]))
    
    freqs_iday = freqs.to(u.day**-1).value
    dfreqs_iday = dfreqs.to(u.day**-1).value
    
    T = (np.max(tm)-np.min(tm))*u.day
    n_nupeak = int(np.ceil((2*T*(dnu+w_dnu)*u.uHz).decompose()))
    
    lls = np.zeros((n_dnu, n_numax))
    ll_nupeak = np.zeros((n_dnu, n_numax, n_nupeak))
    
    # nan grid points outside the search area
    xv, yv = np.meshgrid(dfreqs, freqs, indexing='ij')
    dnu_grid = dnu_numax(freqs.value)
    ind_outside = np.abs(xv.value-dnu_grid)>2.5
    lls[ind_outside] = np.nan
    
    ncall = np.sum(np.isfinite(lls))*n_nupeak
    time = ncall * 2.4 * 1e-4
    print(ncall, time)
    
    for i, f in enumerate(freqs_iday[:]):
        if filter:
            # these values are in uHz
            numax_now = freqs[i].value
            gamma = 0.66 * numax_now**0.88
            band_width = 2
            fmin = max(40, numax_now - band_width*gamma)
            fmax = numax_now + band_width*gamma
            #print(fmin, fmax)
            #fmin, fmax = 40, 50
            
            fm_ = bandpass_filter(fm, fmin, fmax, fs)
        else:
            fm_ = fm
        
        for j, df in enumerate(dfreqs_iday[:]):
            if np.isfinite(lls[j,i]):
                if mg=='none':
                    lls[j, i] = ln_profile_like_K_freqs(tm, fm_, ivar, f, df, K=K0)
                else:
                    # grid in nupeak
                    ddf = 2*df/n_nupeak
                    freqs_peak_iday = np.arange(f-df+0.5*ddf, f+df, ddf)
                    lgrid = np.zeros(n_nupeak)
                    
                    # evaluate likelihoods on the nupeak grid
                    for l, fp in enumerate(freqs_peak_iday):
                        lgrid[l] = ln_profile_like_K_freqs(tm, fm_, ivar, fp, df, K=K0)
                    ll_nupeak[j, i] = lgrid
                    
                    # marginalize over the nupeak grid
                    if mg=='max':
                        lls[j, i] = np.max(lgrid)
                    else:
                        mg = 'mean'
                        lls[j, i] = logsumexp(lgrid) - np.log(np.size(lgrid))
                #print(lls[j, i])
    np.savez('../data/lhood_tic.{:09d}_k.{:d}_mg.{:s}_filter.{:d}'.format(tin['TIC'][i0], K0, mg, filter), freqs=freqs, dfreqs=dfreqs, lls=lls, ll_nupeak=ll_nupeak)

def plot_search(i0=0, K0=5, mg='mean', filter=False):
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    fin = np.load('../data/lhood_tic.{:09d}_k.{:d}_mg.{:s}_filter.{:d}.npz'.format(tin['TIC'][i0], K0, mg, filter))
    freqs = fin['freqs']*u.uHz
    dfreqs = fin['dfreqs']*u.uHz
    lls = fin['lls']
    
    plt.close()
    plt.figure(figsize=(8,7))
    
    print(np.nanmax(lls))
    plt.imshow(lls, origin='lower', extent=[freqs[0].value, freqs[-1].value, dfreqs[0].value, dfreqs[-1].value], aspect='auto', cmap='viridis')
    
    numax_err = np.sqrt(tin['Stnumax']**2 + tin['Synumax']**2)[i0]
    dnu_err = np.sqrt(tin['StDelNu']**2 + tin['SyDelNu']**2)[i0]
    
    pe = mpl.patches.Ellipse([tin['numax'][i0], tin['Delnu'][i0]], width=2*numax_err, height=2*dnu_err, facecolor='none', edgecolor='r', lw=2, zorder=10)
    plt.gca().add_patch(pe)

    plt.xlabel('$\\nu_{max}$ [$\mu$Hz]')
    plt.ylabel('$\Delta\\nu$ [$\mu$Hz]')

    plt.colorbar(label='ln Likelihood')

    plt.tight_layout()
    plt.savefig('../plots/lhood_tic.{:09d}_k.{:d}_mg.{:s}_filter.{:d}.png'.format(tin['TIC'][i0], K0, mg, filter))


# marginalized likelihood (L2)

def lnlike_0(ts, ys, yivars, deltanu, nupeak, K, numax, Lambda0, H, Gamma):
    """Log Likelihood for a comb of K frequencies centered on nupeak and separated by deltanu (solves for the amplitudes)"""
    
    assert len(ts) == len(ys)
    
    halfK = (K - 1) // 2
    thisK = 2 * halfK + 1
    
    M = np.zeros((len(ts), 2 * thisK + 1))
    M[:, 0] = 1.
    for k in range(thisK):
        f = nupeak - halfK * deltanu + k * deltanu
        M[:, 2 * k + 1] = np.cos(2. * np.pi * f * ts)
        M[:, 2 * k + 2] = np.sin(2. * np.pi * f * ts)
    
    theta = np.linalg.solve(np.dot(M.T * yivars, M), np.dot(M.T * yivars, ys))
    residual = ys - np.dot(M, theta)
    
    return -0.5 * np.sum(yivars * residual ** 2) -0.5 * np.sum(np.log(2*np.pi * yivars))

def lnlike_1(ts, ys, yivars, deltanu, nupeak, K, numax, Lambda0, H, Gamma):
    """Log Likelihood for a comb of K frequencies centered on nupeak and separated by deltanu and amplitudes determined by the bell parameters numax, Lambda0, H and Gamma"""
    
    assert len(ts) == len(ys)
    
    halfK = (K - 1) // 2
    thisK = 2 * halfK + 1
    
    M = np.zeros((len(ts), 2 * thisK + 1))
    M[:, 0] = 1.
    for k in range(thisK):
        f = nupeak - halfK * deltanu + k * deltanu
        M[:, 2 * k + 1] = np.cos(2. * np.pi * f * ts)
        M[:, 2 * k + 2] = np.sin(2. * np.pi * f * ts)
    
    L = np.zeros(2*thisK + 1)
    L[0] = Lambda0
    for k in range(thisK):
        f = nupeak - halfK * deltanu + k * deltanu
        h = H * np.exp(-(f - numax)**2/Gamma**2)
        L[2*k + 1] = h
        L[2*k + 2] = h
    #L = np.diag(L)
    
    B = np.diag(yivars**-1) + np.dot(M*L, M.T)
    
    
    # inverse
    I = np.eye(len(yivars))
    #t1 = time.time()*u.s
    #Binv = np.linalg.inv(B)
    #t2 = time.time()*u.s
    ##print(Binv)
    #print(np.dot(Binv,B), np.allclose(np.dot(Binv,B), I, atol=1e-5))
    
    t1 = time.time()*u.s
    Binv_fast = lemma_inv(B, M, yivars, L)
    t2 = time.time()*u.s
    print(t2-t1)

    #print(Binv_fast)
    #print(np.allclose(Binv, Binv_fast, atol=1e2))
    #print(np.dot(Binv_fast,B), np.allclose(np.dot(Binv_fast,B), I))
    
    #print(Binv - Binv_fast)
    #print(np.linalg.cond(B))
    #print(np.linalg.cond(np.dot(M.T * yivars**-1, M)))
    
    # determinant
    #t1 = time.time()*u.s
    #s, logdetB = np.linalg.slogdet(B)
    #assert s==1, 'det B should be positive'
    
    t2 = time.time()*u.s
    logdetB_fast = lemma_logdet(B, M, yivars, L)
    t3 = time.time()*u.s
    print((t3-t2))
    
    #print(logdetB, logdetB_fast)
    #print(np.allclose(logdetB, logdetB_fast))
    
    
    
    lnl = -0.5 * np.dot(np.dot(ys,Binv_fast),ys) - 0.5*logdetB_fast - 0.5*np.log(2*np.pi)
    
    return lnl

def lemma_inv(B, M, yivars, L):
    """Use matrix inversion lemma to calculate the inverse of B = C + M L M.T"""
    
    Cinv = np.diag(yivars)
    MCinv = M.T*yivars
    
    S = np.diag(L**-1) + np.dot(M.T * yivars, M)
    Sinv = np.linalg.inv(S)
    
    D = np.linalg.multi_dot([MCinv.T, Sinv, MCinv])
    
    #D_ = Cinv.dot(M).dot(Sinv).dot(M.T).dot(Cinv)
    #D_ = np.linalg.multi_dot([Cinv, M, Sinv, M.T, Cinv])
    #print(np.allclose(D, D_, rtol=1e-3, atol=1e-3))
    
    Binv = Cinv - D
    
    return Binv

def lemma_logdet(B, M, yivars, L):
    """Use matrix determinant lemma to calculate the log determinant of B = C + M L M.T"""
    
    S = np.dot(M.T * yivars, M) * L
    d = np.einsum('ii->i', S)
    d += 1
    
    s, logdetS = np.linalg.slogdet(S)
    assert s==1
    
    logdetC_fast = np.sum(np.log(yivars**-1))
    #s, logdetC = np.linalg.slogdet(np.diag(yivars**-1))
    #assert s==1
    #print(logdetC, logdetC_fast, np.allclose(logdetC, logdetC_fast))
    
    logdetB = logdetS + logdetC_fast
    
    return logdetB

def test_l1(i0=0, K=5):
    """"""
    tin = Table.read('../data/aguirre_1sec.fits')
    
    # load lightcurve
    t = Table(fits.getdata(tin['fname'][i0], ignore_missing_end=True))
    tm = t['TIME']
    fs = 0.5/(tm[1]-tm[0])
    fm = t['PDCSAP_FLUX']
    fm = (fm - np.nanmean(fm))/np.nanmean(fm)
    
    
    ind_finite = np.isfinite(fm)
    tm = tm[ind_finite]
    fm = fm[ind_finite]
    
    #all_ind = np.arange(len(fm), dtype=int)
    #np.random.seed(183)
    #ind_rand = np.random.choice(all_ind, size=10000)
    ##ind_rand = np.arange(100, dtype=int)
    #tm = tm[ind_rand]
    #fm = fm[ind_rand]
    
    ivar = (np.ones_like(fm)*1e-8)**-1
    print(np.size(ivar))
    
    dnu_uhz = 10*u.uHz
    nupeak_uhz = 60*u.uHz
    numax_uhz = 60*u.uHz
    Lambda0 = 0
    H = 10
    Gamma_uhz = 15*u.uHz
    
    dnu = dnu_uhz.to(u.day**-1).value
    nupeak = nupeak_uhz.to(u.day**-1).value
    numax = numax_uhz.to(u.day**-1).value
    Gamma = Gamma_uhz.to(u.day**-1).value
    
    t1 = time.time()*u.s
    l = lnlike_1(tm, fm, ivar, dnu, nupeak, K, numax, Lambda0, H, Gamma)
    t2 = time.time()*u.s
    print(H, l, t2-t1)
    
    #for H in np.logspace(-1,3,10):
        #l = lnlike_1(tm, fm, ivar, dnu, nupeak, K, numax, Lambda0, H, Gamma)
        #print(H, l)
