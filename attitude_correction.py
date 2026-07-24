import math as m
import numpy as np
import scipy as sp

from obspy.core import UTCDateTime, read, Stream, Trace

"""
The secant function

"""
def sec(x):
    return 1/m.cos(x)

"""
The implementation of steps 1 - 4 from the notebook. The function is basically
a python translation of the method presented in Lin et al. 2010, BSSA

The units are 
m/s/s   for acceleration
rad/s   for rotation rate
Hz      for the cut-off frequencies and for the sampling rate

INPUT:
    TAxB: numpy array
        transverse acceleration x direction in the body frame
    TAyB: numpy array
        transverse acceleration y direction in the body frame
    TAzB: numpy array
        transverse acceleration z direction in the body frame
    RVxB: numpy array
        rotational velocity around x direction in the body frame
    RVyB: numpy array
        rotational velocity around y direction in the body frame
    RVzB: numpy array
        rotational velocity around z direction in the body frame
    sps: float
        sampling rate in samples per second
    one_g: float
        earth gravitational acceleration in m/s/s
    fmin: float
        high-pass cut-off frequency
    fmax: float
        low-pass cut-off frequency

OUTPUT:
    Tuple (TAxG, TAyG, TAzG, TVxG, TVyG, TVzG, TDxG, TDyG, TDzG, EAxG, EAyG, EAzG)
    TAxG: numpy array
        transverse acceleration x direction in the global frame
    TAyG: numpy array
        transverse acceleration y direction in the global frame
    TAzG: numpy array
        transverse acceleration z direction in the global frame
    TVxG: numpy array
        transverse velocity x direction in the global frame
    TVyG: numpy array
        transverse velocity y direction in the global frame
    TVzG: numpy array
        transverse velocity z direction in the global frame
    TDxG: numpy array
        transverse displacement x direction in the global frame
    TDyG: numpy array
        transverse displacement y direction in the global frame
    TDzG: numpy array
        transverse displacement z direction in the global frame
    EAxG: numpy array
        Euler angle around x direction in the global frame
    EAyG: numpy array
        Euler angle around y direction in the global frame
    EAzG: numpy array
        Euler angle around z direction in the global frame
"""

def Attitude(TAxB, TAyB, TAzB, RVxB, RVyB, RVzB, sps, one_g, fmin, fmax):
####################################################################################################
    
    NN = len(RVxB)				# Length of ORIGINAL data vectors
    dt = 1./sps
  
    if NN != len(RVyB) or NN != len(RVzB) or NN != len(TAxB) or NN != len(TAyB) or NN != len(TAzB):
        print('error: Input vectors must be of equal length!')

    EAx0G = 0
    EAy0G = 0
    EAz0G = 0
     
    ##################################################
    # Pad inputs with a leading zero (length NN+1;
    # thus, ignore first output value) and create
    # output vectors
    
    pTAxB = np.zeros(NN+1)
    pTAxB[1:] = TAxB                    # Input acceleration in {B} ("p" means padded)
    pTAyB = np.zeros(NN+1)
    pTAyB[1:] = TAyB                    # Input acceleration in {B} ("p" means padded)
    pTAzB = np.zeros(NN+1)
    pTAzB[1:] = TAzB                    # Input acceleration in {B} ("p" means padded)
    
    pRVxB = np.zeros(NN+1)
    pRVxB[1:] = RVxB                    # Input rotation rate in {B} ("p" means padded)
    pRVyB = np.zeros(NN+1)
    pRVyB[1:] = RVyB                    # Input rotation rate in {B} ("p" means padded)
    pRVzB = np.zeros(NN+1)
    pRVzB[1:] = RVzB                    # Input rotation rate in {B} ("p" means padded)
    
    pEAxG = np.ones(NN+1)
    pEAxG[0] = EAx0G	# Initialize Euler angles (output; padded)
    pEAyG = np.ones(NN+1)
    pEAyG[0] = EAy0G
    pEAzG = np.ones(NN+1)
    pEAzG[0] = EAz0G
    
    pTVxB = np.zeros(NN+1)	# Starts motionless in {B} (padded)
    pTVyB = np.zeros(NN+1)
    pTVzB = np.zeros(NN+1)
    
    pTVxG = np.ones(NN+1)	# Output velocity in {G} (output; padded)
    pTVyG = np.ones(NN+1)
    pTVzG = np.ones(NN+1)
    
    
    ##################################################
    # Loop over time
    
    for ii in range(NN):
    
        ##########################################
        # Values used below repeatedly and buffered inputs,
        # all for time step (ii) and used to generate
        # outputs at both time steps (ii) and (ii+1)
    
        SinXg  = np.sin(pEAxG[ii])		# Used repeatedly below
        CosXg  = np.cos(pEAxG[ii])
    
        SinYg  = np.sin(pEAyG[ii])		# Ditto ...
        CosYg  = np.cos(pEAyG[ii])
        TanYg  = np.tan(pEAyG[ii])
        SecYg  = sec(pEAyG[ii])
    
        SinZg  = np.sin(pEAzG[ii])		# Ditto ...
        CosZg  = np.cos(pEAzG[ii])
    
        TpRVx  = pRVxB[ii]			# Buffers to avoid overwriting inputs
        TpRVy  = pRVyB[ii]			#    during computations; also faster
        TpRVz  = pRVzB[ii]			#    recall ("T" means temporary)
    
        TpTVxB = pTVxB[ii]			# Ditto ...
        TpTVyB = pTVyB[ii]
        TpTVzB = pTVzB[ii]
    
        TpEAx  = pEAxG[ii]			# Ditto ...
        TpEAy  = pEAyG[ii]
        TpEAz  = pEAzG[ii]
    
    
        ##########################################
        # Step (1)-- Eqn (10) from Lin et al. 20210: ATTITUDE EQUATION; find the
        # orientation of {B} within {G} (output for time
        # step (ii+1))
    
        pEAxG[ii+1] = pEAxG[ii] + (dt * ( TpRVx + (SinXg * TanYg * TpRVy) + (CosXg * TanYg * TpRVz) ))
    
        pEAyG[ii+1] = pEAyG[ii] + (dt * ((CosXg * TpRVy) - (SinXg * TpRVz) ))
    
        pEAzG[ii+1] = pEAzG[ii] + (dt * ((SinXg * SecYg * TpRVy) + (CosXg * SecYg * TpRVz) ))

        ########################################## 
        # Step (2)-- Eqn (15) from Lin et al. 20210: ATTITUDE-CORRECTION EQUATION -- to
        # correct the input velocities for tilt (gravity) 
        pTVxB[ii+1] = TpTVxB + (dt * ( pTAxB[ii] + (one_g * SinYg) ))
    
        pTVyB[ii+1] = TpTVyB + (dt * ( pTAyB[ii] - (one_g * CosYg * SinXg) ))
    
        pTVzB[ii+1] = TpTVzB + (dt * ( pTAzB[ii] + (one_g * (1 - CosYg * CosXg)) ))
        ########################################## 
        # Step (3)-- Eqn (4) from Lin et al. 20210: TRANSFORM {B} velocities to
        # {G} velocities (NOTE:  output for time step (ii))
    
        pTVxG[ii] = TpTVxB * (CosZg * CosYg) + TpTVyB * (CosZg * SinYg * SinXg - SinZg * CosXg) + TpTVzB * (CosZg * SinYg * CosXg + SinZg * SinXg)
    
        pTVyG[ii] = TpTVxB * (SinZg * CosYg) + TpTVyB * (SinZg * SinYg * SinXg + CosZg * CosXg) + TpTVzB * (SinZg * SinYg * CosXg - CosZg * SinXg)
    
        pTVzG[ii] = TpTVxB * (-SinYg       ) + TpTVyB * (CosYg * SinXg                        ) + TpTVzB * (CosYg * CosXg                        )

    #########################
    # Since steps (1) and (2) are creating values for
    # time step (ii+1) from values at time step (ii) BUT
    # step (3) is only creating values at time step (ii),
    # so the value of pTV[xyz]B(ii+1) is computed here:
    
    TpTVxB = pTVxB[NN]
    TpTVyB = pTVyB[NN]
    TpTVzB = pTVzB[NN]
    
    SinXg  = np.sin(pEAxG[NN])
    CosXg  = np.cos(pEAxG[NN])
    
    SinYg  = np.sin(pEAyG[NN])
    CosYg  = np.cos(pEAyG[NN])
    
    SinZg  = np.sin(pEAzG[NN])
    CosZg  = np.cos(pEAzG[NN])
    
    pTVxG[NN] = TpTVxB * (CosZg * CosYg) + TpTVyB * (CosZg * SinYg * SinXg - SinZg * CosXg) + TpTVzB * (CosZg * SinYg * CosXg + SinZg * SinXg)
    
    pTVyG[NN] = TpTVxB * (SinZg * CosYg) + TpTVyB * (SinZg * SinYg * SinXg + CosZg * CosXg) + TpTVzB * (SinZg * SinYg * CosXg - CosZg * SinXg)
    
    pTVzG[NN] = TpTVxB * (-SinYg       ) + TpTVyB * (CosYg * SinXg                        ) + TpTVzB * (CosYg * CosXg                        )
    
    
    #########################
    # Reasonably selected records should have zero initial
    # translational velocity, so force this result
    
    tr_pTVxG = Trace(data=pTVxG)
    tr_pTVxG.stats.sampling_rate = sps
    st_ptVxG = Stream(traces=tr_pTVxG)

    tr_pTVyG = Trace(data=pTVyG)
    tr_pTVyG.stats.sampling_rate = sps
    st_ptVyG = Stream(traces=tr_pTVyG)

    tr_pTVzG = Trace(data=pTVzG)
    tr_pTVzG.stats.sampling_rate = sps
    st_ptVzG = Stream(traces=tr_pTVzG)

    
    ##################################################
    # Step (4a), integrate {G} velocities to {G}
    # displacements (does trivial demean by first sample
    
    
    ##################################################
    # Step (4b), differentiate {G} velocities to {G} accelerations (leading
    # and ending samples of the velocity, pTV[xyz]G, are lost in Deriv3(), so
    # replicate end velocity value on input (the leading velocity input sample
    # being a pad already) and replaces beginning zero on output, the latter
    # just for consistency with other front-padded output arrays)
    
    for tr in st_ptVxG:
        mean = np.mean(tr[:250])
        tr.data = tr.data - mean
    st_ptAxG = st_ptVxG.copy()
    st_ptDxG = st_ptVxG.copy()
    st_ptAxG.differentiate()
    st_ptDxG.integrate()
    for tr in st_ptDxG:
        mean = np.mean(tr[:250])
        tr.data = tr.data - mean

    for tr in st_ptVyG:
        mean = np.mean(tr[:250])
        tr.data = tr.data - mean
    st_ptAyG = st_ptVyG.copy()
    st_ptDyG = st_ptVyG.copy()
    st_ptAyG.differentiate()
    st_ptDyG.integrate()
    for tr in st_ptDyG:
        mean = np.mean(tr[:250])
        tr.data = tr.data - mean
    
    for tr in st_ptVzG:
        mean = np.mean(tr[:250])
        tr.data = tr.data - mean
    st_ptAzG = st_ptVzG.copy()
    st_ptDzG = st_ptVzG.copy()
    st_ptAzG.differentiate()
    st_ptDzG.integrate()
    for tr in st_ptDzG:
        mean = np.mean(tr[:250])
        tr.data = tr.data - mean

    #st_ptDzG.taper(0.01)
    st_ptDzG.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    #st_ptDyG.taper(0.01)
    st_ptDyG.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    #st_ptDyG.taper(0.01)
    st_ptDxG.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    
    pTAxG = st_ptAxG[0].data
    pTAyG = st_ptAyG[0].data
    pTAzG = st_ptAzG[0].data

    pTVxG = st_ptVxG[0].data
    pTVyG = st_ptVyG[0].data
    pTVzG = st_ptVzG[0].data
    
    pTDxG = st_ptDxG[0].data
    pTDyG = st_ptDyG[0].data
    pTDzG = st_ptDzG[0].data
    
    ##################################################
    # Remove the leading zero pad (the trailing pad
    # for accelerations fell away in Deriv3())
    
    TAxG = pTAxG[1:NN]
    TAyG = pTAyG[1:NN]
    TAzG = pTAzG[1:NN]
    
    TVxG = pTVxG[1:NN]
    TVyG = pTVyG[1:NN]
    TVzG = pTVzG[1:NN]
    
    TDxG = pTDxG[1:NN]
    TDyG = pTDyG[1:NN]
    TDzG = pTDzG[1:NN]
    
    EAxG = pEAxG[1:NN]
    EAyG = pEAyG[1:NN]
    EAzG = pEAzG[1:NN] 

    return (TAxG, TAyG, TAzG, TVxG, TVyG, TVzG, TDxG, TDyG, TDzG, EAxG, EAyG, EAzG)

