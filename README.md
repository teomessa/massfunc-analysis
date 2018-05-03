# massfunc-analysis
Contains routines for the analysis of mass functions


Routines uploaded:

- **CMF_MCsampling:**  
Performs the Monte Carlo sampling of the mass function.  
Takes masses from a file of which the first column is read as the list of cluster masses (the file must contain 1 or 2 columns, masses can be either in lin or log space). File name and other options can be specified in the input file.  
The simulated mass functions contain the same number of sources as the input file. Either a simple or a truncated power-law can be used as the PDF from which to extract the masses.  
The median of the simulated functions is compared to the input mass distribution via both a Kolmogorov-Smirnov and an Anderson-Darling test.  
A plot is produced showing the cumulative input mass distribution along with the results of simulations. 

- **CMF_Bayes:**  
Performs a fit of the mass function, using a Bayesian approach.
Takes masses from a file of which the first column is read as the list of cluster masses (the file must contain 1 or 2 columns, masses can be either in lin or log space). File name and other options can be specified in the input file.  
**N.B.: In order to work the code is importing the package emcee: http://dfm.io/emcee/current/**.
**N.B.2: The code needs some improvement (e.g. only a flat prior is now contemplated, layout and outputs should be improved). But the main code is perfectly working**

