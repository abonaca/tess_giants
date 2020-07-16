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
    ayivar = A.T * yivars
    m = np.dot(ayivar, A)
    cond = np.linalg.cond(m)
    
    Asolved = np.linalg.solve(m, np.dot(ayivar, ys))
    yp = np.dot(A, Asolved)
    resid = ys - yp
    chi2 = -0.5 * np.sum(yivars * resid ** 2)
    
    return (Asolved, cond, resid, chi2)

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

def multi_comparison(i0=20, fz=1, voff=0):
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
    
    for i in range(2):
        plt.sca(ax[0][i])
        plt.title('{:s} {:d}'.format(sources[i], tic), fontsize='medium')
        
        plt.sca(ax[1][i])
        plt.xlabel('$\\nu$ [$\mu$Hz]')
        
        plt.sca(ax[i][0])
        plt.ylabel('$\Delta_{\\nu}$ [$\mu$Hz]')
    
    plt.tight_layout()
    plt.savefig('../plots/mock_comparison_{:09d}_fz{:.1f}_voff{:04d}.png'.format(tic, fz, voff))

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

