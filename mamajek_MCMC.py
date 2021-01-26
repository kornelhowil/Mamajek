from matplotlib import use as mpluse
from matplotlib import rc
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import math, sys, os, time
from astropy import units as u
from astropy.coordinates import SkyCoord
from gal_jac_dens import *
from parsubs import *
from Mamajek import *
import weightedstats as ws
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

fun = initialize_interp()
rc('text', usetex=False) 


#-------------------------------------------------------------------------------------
################################ INPUT AND CONSTANTS #################################
#-------------------------------------------------------------------------------------

fname_samples =  "BLG196.5-CLEAN_CUT_1_472.91439489720904.npy" 
# cords
alpha = 270.84187
delta = -29.25871
# plotname
plotname = "BLG196.5_sol1"
# extinction
muRC = 14.526
AI =  0.956
# t0par
t0par = 4818 
# number of iterations
L = 5e4 
# some fixed distance values
DSest = 8000 
DSmin = 7999 
DSmax =  8001 
DSmin /= 1000.
DSmax /= 1000.  
# idk what is it, but needed
masspower=1.00 
# DEF: 0,4.26,0,4.26; 4.0, 2.55, 0, 2.55
mu_b_x, sig_b_x = 0.0 , 4.26  
mu_b_y, sig_b_y = 0.0 , 4.26 
mu_d_x, sig_d_x = 0.0 , 2.55
mu_d_y, sig_d_y = 4.0 , 2.55 
# const
deg2rad = 3.14159265359/180.
KAPPA = 8.144
const = 5./KAPPA #scaling for efficiency const*teH/piE
# velocities
vsun_n,vsun_e = get_sun_v(alpha, delta, t0par)
vsun_n_kms=((vsun_n*u.AU).to(u.km).value)/86400.
vsun_e_kms=((vsun_e*u.AU).to(u.km).value)/86400.


eq2gal=SkyCoord(ra=alpha*u.degree, dec=delta*u.degree, frame='icrs') # in degrees, get the l,b
gall=eq2gal.galactic.l.degree
galb=eq2gal.galactic.b.degree

## READING THE FILE
print("Reading input file: ", fname_samples)
data = np.load(fname_samples)

#-------------------------------------------------------------------------------------
################################## MULTIPROCESSING ###################################
#-------------------------------------------------------------------------------------

print("--- Starting Multiprocessing ---")

def MultiProcessingLoop(iteration):

    global count
    np.random.seed()

    t0 = data[iteration][0]
    tE = data[iteration][1]
    u0 = data[iteration][2]
    piEN = data[iteration][3]
    piEE = data[iteration][4]
    I0 = data[iteration][5]
    fs = data[iteration][6]

    piE = np.sqrt(piEN*piEN+piEE*piEE)
    # Correcting tE to heliocentric view
    oneovertEH_n = -vsun_n*piE + piEN/(tE*piE)
    oneovertEH_e = -vsun_e*piE + piEE/(tE*piE)
    oneovertEH = np.sqrt(oneovertEH_n**2+oneovertEH_e**2)
    teH = 1./oneovertEH
    # Requiring the same angle as piEE-piEN - equatorial coords, but mu is in galactic.
    # Converting piEE,piEN to galactic:
    # In galactic coordinates, the PA or equatorial north pole, approximate angle
    northPA=60.*deg2rad
    piEN_g=piEN*np.cos(northPA)-piEE*np.sin(northPA) # north galactic coord of piE
    piEE_g=piEN*np.sin(northPA)+piEE*np.cos(northPA) # east  galactic coord of piE
    piEvec = [piEE_g, piEN_g]
    # What accuracy to require?
    accuracyalpha = 15 # deg + and -
    # How long until right angle is found
    alphawait=1000
    alphacount=0
    angle=10000.
    # Random murel from distribution:
    while ((np.abs(angle)>accuracyalpha) and (alphacount<alphawait)):
        mu1x = np.random.uniform(mu_d_x-15., mu_d_x+15.) 
        mu1y = np.random.uniform(mu_d_y-15., mu_d_y+15.) 
        mu2x = np.random.uniform(mu_b_x-15., mu_b_x+15.) 
        mu2y = np.random.uniform(mu_b_y-15., mu_b_y+15.) 
        murelvec = [mu1x-mu2x, mu1y-mu2y]
        # Angular difference between two vectors, assuring they have the same direction:
        angle = np.degrees(np.math.atan2(np.linalg.det([piEvec,murelvec]),np.dot(piEvec,murelvec)))
        alphacount+=1

    murel = np.sqrt((mu1x-mu2x) ** 2 + (mu1y - mu2y) ** 2)
    # SOURCE DISTANCE - randomising very broadly - it will be weighted later with the Galaxy
    DS = np.random.uniform(DSmin, DSmax) 
    # Weight: not normalised
    w_gal = get_gal_jac(alpha,delta, piEN, piEE, murel, teH, (1./DS), t0par, vsun_n_kms,vsun_e_kms,gall,galb, masspower)
    w_gaia = 1.0 
    # Mass
    mass = murel * ((teH / 365.25) / piE) / KAPPA
    # Lens distance
    distance = 1. / (murel * (teH / 365.25) * piE + 1./DS)
    # Blend brightness from blending parameter (fs); fs>1 ("negative blending") is allowed, but not too much (<~1.4)
    fblend = 1 - fs
    if (fs < 1): blend = I0 - 2.5 * np.log10(fblend)
    else: blend = 22.
    if blend > 22: blend=22.
    # Brightness of the lens from its mass - now using Mamajek table
    Ilens = get_MS_mag(fun,mass, distance, AI)
    # Brightness of the source star from fs
    isource= I0 - 2.5 * np.log10(fs)
    if (fs < 0): isource = 24.

    return [w_gal, w_gaia, mass, distance, blend, Ilens, isource]

start_time = time.time()

results = []
if __name__ == '__main__':
    set_start_method('fork')
    iterations = np.random.randint(len(data),size=int(L))
    with Pool() as pool:
        with tqdm(total=int(L)) as pbar:
            for i in pool.imap_unordered(MultiProcessingLoop, iterations):
                results.append(i)
                pbar.update()

print("--- Multiprocessing took %s seconds ---" % (time.time() - start_time))

results_t = np.array(results).T 
#0 - w_gal, 1 - w_gaia, 2 - mass, 3 - distance, 4 - blend, 5 - Ilens, 6 - isource
storewgal = results_t[0]
storewgaia=results_t[1]
ML = results_t[2]
DL = results_t[3]
Ibl = results_t[4]
IL = results_t[5]
IS = results_t[6]
W=[] 
# Normalising weights:
w1 = storewgal/np.sum(storewgal)
w2 = storewgaia/np.sum(storewgaia)
W = w1*w2

#-------------------------------------------------------------------------------------
############################### PLOTS AND OUTPUTS ####################################
#-------------------------------------------------------------------------------------

#            [0,   1,  2, 3,  4]
ZIP=np.array([Ibl, ML, W, DL, IL]).T

# Correct computation of remnant prob
remn = ZIP[ZIP[:,4]<ZIP[:,0]]
nonrem = ZIP[ZIP[:,4]>=ZIP[:,0]]
remnant = np.sum(remn[:,3])
nonremnant = np.sum(nonrem[:,3])

# Computing the values:
bins = np.linspace( -1, 1.5, 200)
bins= np.linspace(0.1, 40, 200)
n, bins0 = np.histogram((ML), bins=bins, weights= W, density=True)
nsum = np.cumsum(n*(bins[1]-bins[0]))
imedian = np.digitize([0.5],nsum)[0]
nmedian = bins[imedian]

perc1minus=bins[np.digitize([0.5-0.341],nsum)[0]]
perc1plus=bins[np.digitize([0.5+0.341],nsum)[0]]

bins = np.linspace(0,8,200)
n,bins0=np.histogram((DL), bins=bins, weights= W, density=True)
nsum=np.cumsum(n*(bins[1]-bins[0]))
imedian=np.digitize([0.5],nsum)[0]
nmedian=bins[imedian]

perc1minus=bins[np.digitize([0.5-0.341],nsum)[0]]
perc1plus=bins[np.digitize([0.5+0.341],nsum)[0]]

# Plots 
fig = plt.figure(figsize=(5,7))

# MASS - DISTANCE
ax1 = plt.axes([0.17,0.6 ,0.80,0.35])
H1, xedges, yedges = np.histogram2d(ZIP[:,1],ZIP[:,3],weights=ZIP[:,2],bins=[10**np.linspace(-1,1.7,100),np.linspace(0,8,100)], normed=True)
map1=plt.pcolormesh(yedges,xedges,H1/np.sum(H1),cmap='jet',norm=LogNorm(1e-5, 1e-2))
ax1.set_ylabel(r'lens mass $[M_\odot]$')
ax1.set_xlabel('lens distance [kpc]')
ax1.set_yscale('log')
ax1.set_xlim(0,8)
ax1.set_ylim(0.1,50)
ax1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
maxid = np.ndarray.argmax(H1)
maxid2d = np.unravel_index(maxid, H1.shape)
cbar=plt.colorbar(map1,ticks=[1e-5,1e-4,1e-3,1e-2])
cbar.ax.set_yticklabels(['-5','-4','-3','-2'])
cbar.set_label("log prob density",labelpad=0)
ax5 = plt.axes([0.17,0.10,0.80,0.35])

# BLEND - LENS
H2, xedges, yedges = np.histogram2d(ZIP[:,0],ZIP[:,4],weights=ZIP[:,2],bins=[np.linspace(15,22,100),np.linspace(15,22,100)], normed=True)
map2 = plt.pcolormesh(yedges,xedges,H2/np.sum(H2),cmap='jet',norm=LogNorm(1e-5, 1e-2), edgecolors='None', linewidth=0)
ax5.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax5.set_xlabel('lens light [mag]')
ax5.set_ylabel('blend light [mag]')
ax5.text(15.2,21.5,'remnant', color='black')
ax5.text(20,16,'MS star', color='r')
ax5.set_xlim(15,22.1)
ax5.set_ylim(22.1,15)
cbar=plt.colorbar(map2,ticks=[1e-5,1e-4,1e-3,1e-2])
cbar.ax.set_yticklabels(['-5','-4','-3','-2'])
cbar.set_label("log prob density",labelpad=0)

plt.plot(np.linspace(1,30,50), np.linspace(1,30,50),'b--')
plt.fill_between(np.linspace(1,30,50),np.linspace(1,30,50),30,alpha=0.1,facecolor='k',edgecolor='None')

plt.savefig(plotname+'.pdf')

# Saving masses and weights
per=np.percentile(DL, [15.8655, 50, 84.1345])
per2=np.percentile(IL, [15.8655, 50, 84.1345])
per3=np.percentile(ML, [15.8655, 50, 84.1345])
#oututing mass range, DL and blend (table 4 in Wyrz16)
#WRONG: these are unweighted!!!!
#print per3[1], per3[2]-per3[1], per3[1]-per3[0], per[1], per[2]-per[1], per[1]-per[0], per2[1], per2[2]-per2[1], per2[1]-per2[0]
#print sys.argv[1],u0solution, pisolution, per[1], per[2]-per[1], per[1]-per[0], per2[1], per2[2]-per2[1], per2[1]-per2[0],per3[1], per3[2]-per3[1], per3[1]-per3[0]
bins=[10**np.linspace(-1,1.7,100),np.linspace(0,8,100)]
H1,xed,yed=np.histogram2d(ML,DL,bins=bins, weights=W, density=True)
maxid = np.ndarray.argmax(H1)
maxid2d = np.unravel_index(maxid, H1.shape)
#print "max.prob mass/dist= ",bins[0][maxid2d[0]],bins[1][maxid2d[1]]
ml1hist,bin=np.histogram(ML,bins=100,weights=W, density=True)
nsum=np.cumsum(ml1hist*(bin[1]-bin[0]))
imedian=np.digitize([0.5],nsum)[0]
implus=np.digitize([0.841345],nsum)[0]
imminus=np.digitize([0.158655],nsum)[0]
dl1hist,bin2=np.histogram(DL,bins=100,weights=W, density=True)
nsum=np.cumsum(dl1hist*(bin2[1]-bin2[0]))
imedian2=np.digitize([0.5],nsum)[0]
implus2=np.digitize([0.841345],nsum)[0]
imminus2=np.digitize([0.158655],nsum)[0]
il1hist,bin3=np.histogram(IL,bins=100,weights=W, density=True)
nsum=np.cumsum(il1hist*(bin3[1]-bin3[0]))
imedian3=np.digitize([0.5],nsum)[0]
implus3=np.digitize([0.841345],nsum)[0]
imminus3=np.digitize([0.158655],nsum)[0]
is1hist,bin4=np.histogram(IS,bins=100,weights=W, density=True)
nsum=np.cumsum(is1hist*(bin4[1]-bin4[0]))
imedian4=np.digitize([0.5],nsum)[0]
implus4=np.digitize([0.841345],nsum)[0]
imminus4=np.digitize([0.158655],nsum)[0]
ib1hist,bin5=np.histogram(Ibl,bins=100,weights=W, density=True)
nsum=np.cumsum(ib1hist*(bin5[1]-bin5[0]))
imedian5=np.digitize([0.5],nsum)[0]
implus5=np.digitize([0.841345],nsum)[0]
imminus5=np.digitize([0.158655],nsum)[0]

print ('weightedMULENS ', bin[imedian], bin[implus]-bin[imedian], bin[imedian]-bin[imminus], bin2[imedian2], bin2[implus2]-bin2[imedian2], bin2[imedian2]-bin2[imminus2])
print (bin3[imedian3], bin3[implus3]-bin3[imedian3], bin3[imedian3]-bin3[imminus3],bin4[imedian4], bin4[implus4]-bin4[imedian4], bin4[imedian4]-bin4[imminus4])
print (bin5[imedian5], bin5[implus5]-bin5[imedian5], bin5[imedian5]-bin5[imminus5])

print ((" prob= %.1f")%(100.*remnant/(nonremnant+remnant)))




