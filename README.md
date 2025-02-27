The code is a "stand alone" version of the white dwarf pipeline currently implemented in 4MOST.
First of all the script uses a random forest classifier trained on 20000 SDSS I - IV white dwarf spectra to classify the input spectrum into 27 possible white dwarf spectral classes.
Only spectra classfied as DA (white dwarfs with only Balmer lines in their spectra) are than passed to a fitting script to calculate Teff and logg for the white dwarf.
This script uses Pier-Emmanuel Tremblay's 1D pure-hydrogen models available at:
https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/

Rather than interpolating the fixed Teff logg grid point models to generate additional models, this fitting routine uses a principle component analysis (PCA) approach to generate additional models at arbitrary Teff and logg. This is an idea developed by Stuart Littlefair (Univeristy of Sheffield). Please see his repository for a complete understanding: https://github.com/StuartLittlefair/wd_emulator

Usage:
Usage:
    python WDpipe.py spec.csv parallax Gaia_G_mag GaiaID

Arguments:
    spec.csv   :Spectrum to analyze saved as a csv file with 3 columns, wavelngth, flux, flux_error
    parallax   :Parallax of the obejct (optional)
    Gaia_G_mag :Gaia G band magnitude of the object (optional)
    GaiaID     :Gaia Source_ID of the object (optional)

Example:
    python WDpipe.py myspec.csv 11.2 18.3 4652527353933233968

If a GaiaID is provided the script will look fir the white dwarf in a reference table from Gentile Fusillo et al. 2021 and use the photometric Teff and logg as an initial guess. This should break the hot/cold solution degeneracy so only one solution is provided in end.

Id the GaiaID is not provided or the value in not found in the reference table the script will calculate both the hot and cold solution on either side of 13,000 K and use the parallax and Gaia G mag to identify the "real" solution between the two.
If parallax and Gaia_G_mag are not provided the script will return both the hot and cold solution.
