import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,glob
import scipy.stats


## -------------------------- USEFUL FUNCTIONS -------------------------------

def read_input_2col(infile):
    #Read in file
    raw_userinput = np.genfromtxt(infile, dtype=None)
    #convert the input matrix to a dictionary for easy access to the inputs
    userinput = dict(zip(raw_userinput[:, 0], raw_userinput[:, 1]))
    #Convert from strings to the true datatypes.
    for key in userinput:
        try: userinput[key] = ast.literal_eval(str(userinput[key]))
        except: userinput[key] = str(userinput[key])
    return userinput

def mass_dist_pl(alfa,min_m,max_m,num):
    norm = (max_m/min_m)**(1.-alfa) - 1.
    mi = min_m*(norm*np.random.uniform(size=num) + 1.)**(1./(1.-alfa))
    return mi

def mass_dist_sch(alpha,xstar,xmin,xmax,nr):
    nrx=int(userinput['BINSPDF'])
    #CALCULATE VALUES of PDF (not normalized)
    f=np.zeros(nrx)
    lxmin=np.log10(xmin)
    lxmax=np.log10(xmax)
    ldx=(lxmax-lxmin)/nrx
    seq=np.zeros(nrx)
    for i in range(0,nrx):seq[i]=i
    lx=(lxmax-lxmin)*seq/(nrx) + lxmin + 0.5*ldx
    for ix in range(0,nrx):
        xtmp=10.**lx[ix]
        f[ix]=xtmp**(-alpha)*np.exp(-xtmp/xstar)
    #calc normalisation
    A=0
    for ix in range(0,nrx):
        dx=10.**(lx[ix]+0.5*ldx)-10.**(lx[ix]-0.5*ldx)
        cum=f[ix]*dx
        A+=cum
    A=1/A
    #CALCULATE NORM + PDF
    PDF=A*f
    #CALCULATE CDF
    CDF=np.zeros(nrx)
    #y0=PDF[0]+0.5*(PDF[0]-PDF[1])
    #y1=PDF[0]
    #ymean=0.5*(y0+y1)
    #dx=10.**(lx[0])-10.**(lx[0]-0.5*ldx)
    dx=10.**(lx[0]+0.5*ldx)-10.**(lx[0]-0.5*ldx)
    CDF[0]=PDF[0]*dx
    for ix in range(1,nrx):
        dx=10.**(lx[ix]+0.5*ldx)-10.**(lx[ix]-0.5*ldx)
        CDF[ix]=CDF[ix-1]+PDF[ix]*dx
    #NOW SAMPLE FROM THE CDF
    ran=np.random.uniform(size=nr)
    xs=np.zeros(nr)
    for im in range(0,nr):
        cmin=np.where(CDF < ran[im])
        cmax=np.where(CDF > ran[im])
        cdfmin=CDF[0]
        if len(cmin[0]) == 0: xs[im]=10.**lx[0]
        elif cmin[0][0] >= 0:
            cdfmin=CDF[cmin[0][np.size(cmin[0])-1]]
            cdfmax=CDF[cmax[0][0]]
            frac=(ran[im]-cdfmin)/(cdfmax-cdfmin)
            xsmin=lx[np.where(CDF == cdfmin)]
            xsmax=lx[np.where(CDF == cdfmax)]
            xs[im]=10.**(frac*(xsmax-xsmin)+xsmin)
    xs=np.asarray(xs)
    return xs



## -------------------------- MAIN CODE STARTING HERE -------------------------------

# INTRO
print '\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'
print '- - - Comparing the observed mass function with simulated ones - - - '
print '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n'
# Location of python code
pydir = os.getcwd()

# Define input file and Verify that input file exists
infile = 'CMF_MCsampling.input'
if os.path.exists(pydir + '/' + infile) == False: sys.exit('\nFile {} could not be found in {} - Quitting now...'.format(infile,pydir))

# Read input file
userinput = read_input_2col(infile)

# Read masses
datamass=np.loadtxt(userinput['CAT_DIR']+'/'+userinput['CAT_NAME'])
if np.size(datamass[0])==2:
    mass=datamass[:,0]
    errmass=datamass[:,1]
elif np.size(datamass[0])==1: mass=datamass
else: sys.exit('Wrong number of columns in the mass file - Quitting...')
if userinput['LOGLIN']=='log': mass = 10**(mass)
mcut=float(userinput['MASSCUT'])
mass=mass[np.where(mass>=mcut)]
mass=mass[np.argsort(mass)]
logmass=np.asarray(map(np.log10,mass))
nobs=len(mass)
print 'Num of Obs clusters: ',nobs


#building the Y axis of the cumulative mass function
#(it only depends on nobs)
yplot=np.zeros(nobs)
for k in range(0,nobs): yplot[k]=1.-(k*1.)/(nobs*1.)

# Simulate the mass functions
# Set parameters
nruns=int(userinput['NRUNS'])
if userinput['MMIN']<=mcut: mmin=mcut
else: mmin=float(userinput['MMIN'])
mmax=float(userinput['MMAX'])
slope=float(userinput['SLOPE'])
if userinput['DISTRIBUTION']=='sch': Mc=float(userinput['MC_SCH'])
# Run it
xmod=np.zeros([nobs,nruns])
print '\nStart simulating mass functions... {} times'.format(nruns)
if userinput['DISTRIBUTION']=='sch':
    for n in range(0,nruns):
        mi=mass_dist_sch(slope,Mc,mmin,mmax,nobs)
        logmi=np.asarray(map(np.log10,mi))
        xmod[:,n]=logmi[np.argsort(logmi)]
elif userinput['DISTRIBUTION']=='pl':
    for n in range(0,nruns):
        mi=mass_dist_pl(slope,mmin,mmax,nobs)
        logmi=np.asarray(map(np.log10,mi))
        xmod[:,n]=logmi[np.argsort(logmi)]
else: sys.exit('Wrong distribution selected for simulations - Quitting...')
print ''

# find the median mass function
xplot,dxplot=np.zeros(nobs),np.zeros(nobs)
x50a,x50b,x50c=np.zeros(nobs),np.zeros(nobs),np.zeros(nobs)
x90a,x90b=np.zeros(nobs),np.zeros(nobs)
x68a,x68b=np.zeros(nobs),np.zeros(nobs)

for k in range(0,nobs):
    xplot[k]=np.median(xmod[k,:])
    dxplot[k]=np.std(xmod[k,:])
    # find interesting percentiles
    # ------50,90 and 68 percentiles ---------
    xtemp=xmod[k,:]
    x50a[k],x50b[k],x50c[k]=np.percentile(xtemp,[25,75,50])
    x90a[k],x90b[k]=np.percentile(xtemp,[5,95])
    x68a[k],x68b[k]=np.percentile(xtemp,[16,84])


# calculating KS and AD tests
mcut_ks_log=np.log10(float(userinput['MASSCUT_KS']))
sample1=logmass[np.where(logmass>=mcut_ks_log)]
sample2=xplot[np.where(xplot>=mcut_ks_log)]
#sample1=10**sample1
#sample2=10**sample2
print 'Running K-S and A-D statistics...'
ksres=scipy.stats.ks_2samp(sample1,sample2)
adres=scipy.stats.anderson_ksamp([sample1,sample2])
print '\tKS test:\tD={},  \tp-val={}'.format(ksres[0],ksres[1])
print '\tAD test:\tA={},  \tp-val={}'.format(adres[0],adres[2])
print ''

# PLOT
#find positions
xtext1=logmass[0]+(logmass[-1]-logmass[0])*0.6
ytext1a=yplot[-1]+(yplot[0]-yplot[-1])*0.6
ytext1b=yplot[-1]+(yplot[0]-yplot[-1])*0.4
ytext1c=yplot[-1]+(yplot[0]-yplot[-1])*0.26
xtext2=logmass[0]+(logmass[-1]-logmass[0])*0.1
ytext2a=yplot[-1]+(yplot[0]-yplot[-1])*0.01
ytext2b=yplot[-1]+(yplot[0]-yplot[-1])*0.005
# plotting data and models
plt.plot(xplot,yplot,ls='-',color='black',lw=1.7)
plt.plot(x50a,yplot,ls='--',color='black',lw=1.7)
plt.plot(x50b,yplot,ls='--',color='black',lw=1.7)
plt.plot(x90a,yplot,ls=':',color='black',lw=1.7)
plt.plot(x90b,yplot,ls=':',color='black',lw=1.7)
plt.plot(logmass,yplot,ls='',marker='o',mew=0,mfc='RoyalBlue',ms=8)
# plotting text
plt.text(xtext1,ytext1a,'{} function'.format(userinput['DISTRIBUTION'].upper()),fontsize=14)
plt.text(xtext1,ytext1b,r'$\beta:\ \ -{}$'.format(userinput['SLOPE']),fontsize=14)
if userinput['DISTRIBUTION']=='sch':plt.text(xtext1,ytext1c,r'$\rm M_c:\ {}\ M_\odot$'.format(userinput['MC_SCH']),fontsize=14)
plt.text(xtext2,ytext2a,'p-val(KS): {}'.format(round(ksres[1],3)),fontsize=14)
plt.text(xtext2,ytext2b,'p-val(AD): {}'.format(round(adres[2],3)),fontsize=14)
# adjusting
plt.ylim(0.8*yplot[-1],1.)
plt.xlim(np.log10(mcut),logmass[-1]+(logmass[-1]-logmass[0])*0.1)
plt.yscale('log')
plt.ylabel(r'$\rm Cumulative\ fraction$',fontsize=16)
plt.xlabel(r'$\rm Log(Mass/M_{\odot})$',fontsize=16)
plt.tight_layout()
plt.savefig('plot_CMF_MCsampling.pdf')



