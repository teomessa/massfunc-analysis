import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter, MaxNLocator
import numpy as np
import pandas as pd
import scipy.integrate
import sys,os,glob
import emcee
import corner


## -------------------------- USEFUL FUNCTIONS -------------------------------

def read_input_2col_old(infile):
    #Read in file
    raw_userinput = np.genfromtxt(infile, dtype=None)
    #convert the input matrix to a dictionary for easy access to the inputs
    userinput = dict(zip(raw_userinput[:, 0], raw_userinput[:, 1]))
    #Convert from strings to the true datatypes.
    for key in userinput:
        try: userinput[key] = ast.literal_eval(str(userinput[key]))
        except: userinput[key] = str(userinput[key])
    return userinput

def read_input_2col(infile):
    cat=np.loadtxt(infile,dtype='str')
    userinput={}
    for key,val in cat: userinput[key]=val
    #Convert from strings to the true datatypes.
    floats_ls=['MASSCUT','SCH_START_BETA','SCH_START_MC','SCH_START_SIGMA','PL_START_BETA','PL_START_SIGMA']
    int_ls=['SCH_NWALK','SCH_STEPS','SCH_BURNIN','PL_NWALK','PL_STEPS','PL_BURNIN']
    for key in floats_ls: userinput[key]=float(userinput[key])
    for key in int_ls: userinput[key]=int(userinput[key])
    return userinput


### FUNCTIONS FOR SCHECHTER FIT

def Pmf(Mi,alpha,Mc):
    return 1e7*(Mi**alpha)*np.exp(-Mi/Mc)

def Pobs():
    return 1.

def norm(alpha,Mc,Mlim):
    int = scipy.integrate.quad(lambda x: (Pmf(x,alpha,Mc)*Pobs()),Mlim,np.inf)
    return int[0]

def prioralphalog(alpha):
    if alpha < -4 or alpha > -1: result=-np.inf
    else: result=0.
    return result

def priorMclog(Mc):
    if Mc < mcut or Mc > 1e9: result=-np.inf
    else: result=0.
    return result

def Pclusterlog(Mlist,alpha,Mc):
    n=len(Mlist)
    sum=0.
    z=norm(alpha,Mc,mcut)
    for i in range(0,n):
        #print(sum)
        sum+=np.log(Pobs())-np.log(z)+alpha*np.log(Mlist[i])-Mlist[i]/Mc+np.log(1e7)
    return sum

def likelihoodlog_sch(theta,Mobs):
    alpha,logMc=theta
    Mc=10**logMc
    lp = prioralphalog(alpha)*priorMclog(Mc)
    if not np.isfinite(lp): return -np.inf
    return Pclusterlog(Mobs,alpha,Mc)+priorMclog(Mc)+prioralphalog(alpha)


### FUNCTIONS FOR PL FIT

def likelihoodlog_pl(theta,Mobs):
    alpha=theta
    lp = prioralphalog(alpha)
    if not np.isfinite(lp): return -np.inf
    return Pclusterpllog(Mobs,alpha)+prioralphalog(alpha)

def Pclusterpllog(Mlist,alpha):
    n=len(Mlist)
    sum=0.
    z=normpl(alpha,mcut)
    for i in range(0,n): sum+=np.log(Pobs())-np.log(z)+alpha*np.log(Mlist[i])+np.log(1e7)
    return sum

def normpl(alpha,Mlim):
    #int = scipy.integrate.quad(lambda x: (Pmfpl(x,alpha)*Pobs()),Mlim,+np.inf)
    int = scipy.integrate.quad(lambda x: (Pmfpl(x,alpha)*Pobs()),Mlim,1e9)
    return int[0]

def Pmfpl(Mi,alpha):
    return 1e7*(Mi**alpha)



## -------------------------- MAIN CODE STARTING HERE -------------------------------

# INTRO
print('\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('- - - Fitting the mass function using Bayesian approach - - - ')
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n')
# Location of python code
pydir = os.getcwd()

# Load input file
if len(sys.argv)<2:
    sys.exit("missing input file in CL\nUSAGE EXAMPLE: python CMF_bayes_py3.py inputfile.input\n")
else: infile = sys.argv[1]

# Define input file and Verify that input file exists
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
#mass=mass[np.argsort(mass)]
#logmass=np.asarray(map(np.log10,mass))
nobs=len(mass)
print('Num of Obs clusters: {}\n'.format(nobs))


# --- --- --- SCH FIT --- --- ---
if userinput['SCH_FLAG'].lower()=='yes':
 ndim, nwalkers = 2, int(userinput['SCH_NWALK'])
 burnin=int(userinput['SCH_BURNIN'])
 beta_start=float(userinput['SCH_START_BETA'])
 Mc_start=np.log10(float(userinput['SCH_START_MC']))
 xstart = np.asarray([beta_start,Mc_start])
 pos = [xstart + float(userinput['SCH_START_SIGMA'])*np.random.randn(ndim) for i in range(nwalkers)]
 sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihoodlog_sch, args=[mass])
 print('Running the chain for SCH...')
 sampler.run_mcmc(pos, int(userinput['SCH_STEPS']))
 print('\t-> run completed')

 # plot chain
 fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
 axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
 axes[0].yaxis.set_major_locator(MaxNLocator(5))
 axes[0].axvline(burnin, color="DodgerBlue", lw=2)
 axes[0].set_ylabel(r"$\beta$")
 axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
 axes[1].yaxis.set_major_locator(MaxNLocator(5))
 axes[1].axvline(burnin, color="DodgerBlue", lw=2)
 axes[1].set_ylabel(r"$\rm{log}(Mc/M_\odot)$")
 fig.tight_layout(h_pad=0.0)
 fig.savefig("plot_CMF_bayes_schchain.pdf")
 print('\t-> chain plotted')

 # plot posterior
 samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
 fig = corner.corner(samples, labels=[r"$\beta$", r"$\rm{log}(Mc/M_\odot)$"])
 fig.savefig("plot_CMF_bayes_schpost.pdf")
 print('\t-> posterior plotted')
 
 # save samples?
 tab_samples = pd.DataFrame()
 tab_samples['alpha'],tab_samples['logMc']=samples[:,0],samples[:,1]
 outfile = 'chain_sch_file.txt'
 tab_samples.to_csv(outfile,sep='\t',float_format='%.5f')

 # bin the posterior to get maximum
 # to be revised
 """
 x,y=[],[]
 for ii in range(0,len(samples)):
    x.append(samples[ii][0])
    y.append(samples[ii][1])
#x,y=np.asarray(x),np.asarray(y)
 lowx,higx=np.min(x),np.max(x)
 lowy,higy=np.min(y),np.max(y)
 xedges = np.arange(round(lowx,1),round(higx,1),0.05)
 yedges = np.arange(round(lowy,1),round(higy,1),0.05)
 H, xedges, yedges = np.histogram2d(np.asarray(x), np.asarray(y), bins=(xedges, yedges))
 maxline,bb=[],[]
 for ii in range(0,len(xedges)-1):
    currentline=H[ii]
    bb.append(np.where(currentline==np.max(currentline))[0][0])
    maxline.append(np.max(currentline))
 maxline=np.asarray(maxline)
 bb=np.asarray(bb)
 aa=np.where(maxline==np.max(maxline))[0][0]
 bbb=bb[aa]
 maxa=(xedges[aa]+xedges[aa+1])/2.
 maxm=(yedges[bbb]+yedges[bbb+1])/2.
 print('The MAX of the 2D distribution is at \ta: {}\tMc: {}'.format(maxa,maxm))
 H = H.T
 plt.clf()
 fig = plt.figure(figsize=(7, 7))
 plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
 fig.tight_layout()
 fig.savefig('asdasd.png')
 """
 
 alpha_mcmc, Mc_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
 print('\t-> Results:')
 print('\t\talpha:    {}  +{} -{}'.format(round(alpha_mcmc[0],2),round(alpha_mcmc[1],2),round(alpha_mcmc[2],2)))
 print('\t\tlog(Mc):   {}  +{} -{}\n'.format(round(Mc_mcmc[0],2),round(Mc_mcmc[1],2),round(Mc_mcmc[2],2)))
else: print('No SCH fit requested! moving on...\n')

# --- --- --- PL FIT --- --- ---
if userinput['PL_FLAG'].lower()=='yes':
 ndim, nwalkers = 1, int(userinput['PL_NWALK'])
 burnin=int(userinput['PL_BURNIN'])
 xstart = float(userinput['PL_START_BETA'])
 pos = [xstart + float(userinput['PL_START_SIGMA'])*np.random.randn(ndim) for i in range(nwalkers)]
 sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihoodlog_pl, args=[mass])
 print('Running the chain for PL...')
 sampler.run_mcmc(pos, int(userinput['PL_STEPS']))
 print('\t-> run completed')

 # plot chain
 plt.clf()
 fig, axes = plt.subplots(1, 1, figsize=(8, 3))
 axes.plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
 axes.yaxis.set_major_locator(MaxNLocator(5))
 axes.axvline(burnin, color="DodgerBlue", lw=2)
 axes.set_ylabel(r"$\beta$")
 fig.tight_layout(h_pad=0.0)
 fig.savefig("plot_CMF_bayes_plchain.pdf")
 print('\t-> chain plotted')

 # plot posterior
 samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
 fig = corner.corner(samples, labels=[r"$\beta$"])
 fig.savefig("plot_CMF_bayes_plpost.pdf")
 print('\t-> posterior plotted')

 # save samples?
 tab_samples = pd.DataFrame()
 tab_samples['alpha']=samples[:,0]
 outfile = 'chain_pl_file.txt'
 tab_samples.to_csv(outfile,sep='\t',float_format='%.5f')

 # plot cumulative mass function with result
 logmass = np.log10(mass[np.argsort(mass)])
 

 # bin the posterior to get maximum
 # to be revised
 """
 xedges = arange(-3,-1,0.1)
 H, xedges = histogram(samples, bins=xedges)
 aaa=where(H==max(H))[0][0]
 maxa=(xedges[aaa]+xedges[aaa+1])/2.
 print('The MAX of the distribution is at \ta: '+str(maxa))
 """
 
 v0,v1,v2 = np.percentile(samples, [16, 50, 84], axis=0)
 alpha_mcmc_pl = [v1[0], v2[0]-v1[0], v1[0]-v0[0]]
 #alpha_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
 print('\t-> Results:')
 print('\t\talpha:    {}  +{} -{}\n'.format(round(alpha_mcmc_pl[0],2),round(alpha_mcmc_pl[1],2),round(alpha_mcmc_pl[2],2)))
else: print('No PL fit requested! moving on...\n')


### ADD PLOTS WITH THE MASS FUNCTION (CUMULATIVE?), THE BEST FIT AND VARIATIONS TO THE FIT (SEE EMCEE WEBSITE AND JOHNSON PAPER AS EXAMPLES?)




