import math as m
import numpy as np
import scipy as sp

from obspy.core import UTCDateTime, read, Stream, Trace
from attitude_correction import Attitude

"""
This function simply applies the attitude correction

INPUTS:
    st: obspy stream object
        must contain 6 traces, 6DoF recordings from an IMU.
    fmin: float
        high-pass cut-off frequency
    fmax: float
        low-pass cut-off frequency
    sps: float
        sampling rate in samples per second
    one_g: float
        earth gravitational acceleration in m/s/s

OUTPUT: tuple of attitude corrected data (see output of Attitude())
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

def apply_att_corr(st, fmin, fmax, sps, one_g):

    #TAxB = st.select(channel='DNZ')[0].data
    #TAyB = st.select(channel='DNY')[0].data
    #TAzB = st.select(channel='DNX')[0].data
    #RVxB = st.select(channel='DJZ')[0].data
    #RVyB = st.select(channel='DJY')[0].data
    #RVzB = st.select(channel='DJX')[0].data

    TAxB = st.select(channel='DNX')[0].data
    TAyB = st.select(channel='DNY')[0].data
    TAzB = st.select(channel='DNZ')[0].data
    RVxB = st.select(channel='DJX')[0].data
    RVyB = st.select(channel='DJY')[0].data
    RVzB = st.select(channel='DJZ')[0].data

    NN   = len(TAxB)

############################
# Remove baselines only from rotation rates

    sMnF = 1                        # s at front to fit

    nMnF = int(sMnF * sps)

# Compute front means

    mnTAfX  = np.mean(TAxB[0:nMnF])             # "f" means "front", i.e., pre-event
    mnTAfY  = np.mean(TAyB[0:nMnF])
    mnTAfZ  = np.mean(TAzB[0:nMnF])

    mnRVfX  = np.mean(RVxB[1:nMnF])             # "f" means "front", i.e., pre-event
    mnRVfY  = np.mean(RVyB[1:nMnF])
    mnRVfZ  = np.mean(RVzB[1:nMnF])

    BlnRVx = mnRVfX
    BlnRVy = mnRVfY
    BlnRVz = mnRVfZ

    RVxB = RVxB - BlnRVx
    RVyB = RVyB - BlnRVy
    RVzB = RVzB - BlnRVz


###############################################
# here we apply the correction
    corr_data = Attitude(TAxB, TAyB, TAzB,
           RVxB, RVyB, RVzB,
           sps, one_g, fmin, fmax)
################################################

    return corr_data
