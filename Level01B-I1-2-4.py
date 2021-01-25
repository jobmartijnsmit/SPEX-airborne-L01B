"""
Created on Mon May 23 16:58:07 2016

@author: martijns

Wrapper routine that calls L0-1B, Issue 1-2, processing functions 

Determines DoLP and Radiance for a specified time interval
Code selects the image and nasdat files to collect and read information from 

Issue 1:   indicates Functionality
Issue 1-1: indicates Issue 1, Version 1

Issue 1-1: Works with all swaths identical data
Issue 1-2: Works with all swaths different, compares well with JR data

Issue 1-2-1: Restoration of 1-2, after fuckup of 1-2. Presently this version works,
but take notice that the proper CKD files are supplied. 

Issue 1-2-2: Trial to include Mueller Matrix demodulation
This appears to work fine, but must be verified to be backward compatible.

Issue 1-2-3: Trial to include masks for LOS pixel selection speed-up
This version also reads new Mij, which contains Mueller Matrix elements Mij as well as AOLP-averaged retardance
and AOLP-averaged polarimetric efficiency


Issue 1-2-4: Demodulate the old way, not with Mueller 
This version uses masks to extract LOS elements
No fine grid on the wavelengths; manywavelengths is ignored
With only 5 wavelengths it is still very slow - about one demodulation per second for all viewports, swaths and wavelengths
So Issue 1-2-3 is with Mueller (?)

Use as follows:
    Open xterm, go to working directory
    $ cd ~/ownCLoud/MyPython/spexairborne/pileline/L01BI1/Work
    Start the code:
    $ python3 Level01b-I1-2-3.py  L01B-I1-2-3-inputfile-nswath31.txt  
    
SCRIPT WORKS  




"""
import h5py
import sys
import os
import numpy as np

from   L01BI1.L01B.ReadL01BInputfile_I2                    import ReadL01BInputfile
from   L01BI1.L01B.CKD_Read.CKD_Read_darkimage             import CKD_Read_darkimage
from   L01BI1.L01B.CKD_Read.CKD_read                       import CKD_read_los
from   L01BI1.L01B.CKD_Read.CKD_read                       import CKD_read_wavelengths, CKD_read_wcoefs
from   L01BI1.L01B.CKD_Read.CKD_read                       import CKD_read_radcal
from   L01BI1.L01B.CKD_Read.CKD_read                       import CKD_read_DoLPcal
from   L01BI1.L01B.CKD_Read.CKD_Read_Swath_Angles          import CKD_Make_FOV_Vectors
from   L01BI1.L01B.CKD_Read.CKD_read                       import CKD_read_Mij_poleff
from   L01BI1.L01B.CKD_Read.CKD_read                       import CKD_read_solarspectrum
from   L01BI1.L01B.BitshiftCorrection                      import BitshiftCorrection
from   L01BI1.L01B.PerUnitTime                             import PerUnitTime
from   L01BI1.L01B.ReadGPSdata                             import decodeGPSword
from   L01BI1.L01B.CorrectDarkcurrent                      import CorrectDarkcurrent
from   L01BI1.L01B.CKD_Read.CKD_Read_Hot_Dead_Pixels       import CKD_Read_Hot_Dead_Pixels
#from   L01BI1.L01B.LOSPixelSelection          import LOSPixelSelection, LOSPixelSelectionMask, LOSPixelSelectionMaskBool
from   L01BI1.L01B.LOSPixelSelection                       import LOSPixelSelection
from   L01BI1.L01B.LOSPixelSelection                       import extractlos_using_mask
from   L01BI1.L01B.sol_flux_at_wavelengths                 import sol_flux_at_wavelengths

from   L01BI1.L01B.Wavelengths                import AlignSpwithSm
from   L01BI1.L01B.Wavelengths                import Assign_Wavelengths

from   L01BI1.L01B.Demodulation               import Demodulate, Demodulate_mij, Demodulate_mij_nl
from   L01BI1.L01B.Setnbitshift               import Setnbitshift
from   L01BI1.L01B.makemasks                  import makemasksfullswath, make_los_masks


import L01BI1.L01B.FindHeartbeat as FH

from   Tools.thingies             import find_nearest
from   Tools.thingies             import linterpolate
from   Tools.thingies             import linterpolate2p
import Tools.mytools as mt
from   Tools.readNASDATfiles import readNASDATfiles, readTimeFromIMAGEfiles
from   Tools.flightfiles import getimagefiles, getnasdatfiles
from   Tools.geolocateviewvectors import geolocateviewvectors
from   L01BI1.Test.CorrectStraylight import CorrectStraylight, importdarkzones

def L01B_Issue2p2(inputfilename):

    print("Hi, you are running the function", sys.argv[0])
    print("In File                         ", os.path.basename(__file__))

    radtodeg = (180/np.arctan(1.0)/4).astype(float)
    # Extract information from  file that determines the flow of the program

    useGPSlonlat = True
    dumphires = False
    straylight_flag = False

    
    # extract all the keyword-value pairs fromm the inputfilenamen and put in directory
    L01B_Parameters       = ReadL01BInputfile(inputfilename)

    # extract parameters from the L01B_Parameters Dictionary
    flightdirectory       = L01B_Parameters["FlightDirectory"]
    #Tstart                = L01B_Parameters["Tstart"]
    #Tend                  = L01B_Parameters["Tend"]
    #nasdatfile            = L01B_Parameters["NASDATFILE"     ]
    NViewports            = L01B_Parameters["NViewport"      ]
    Viewports             = L01B_Parameters["Viewport"       ]

    Wavelengths_dolp      = L01B_Parameters["Wavelength_dolp"]
    NWavelengths_dolp     = len(Wavelengths_dolp)  
    Wavelengths_radi      = L01B_Parameters["Wavelength_radi"]
    NWavelengths_radi     = len(Wavelengths_radi)  

    print("Wavelengths DoLP:",Wavelengths_dolp)
    print("# of Wavelengths:",NWavelengths_dolp)

    rad_only_flag         = L01B_Parameters["Radiance_only"]


    split      = flightdirectory.split("/")
    flightdate = split[len(split)-2]
    print("[main] flightdate: ",flightdate)                   
        
    # CKD Filenames
    NViewports    = len(Viewports)

    # CKD Filenames
    CKD_Dark_File        = L01B_Parameters["CKD_Dark_File"      ]
    CKD_LOS_File         = L01B_Parameters["CKD_LOS_File"       ]
    CKD_Wavelength_File  = L01B_Parameters["CKD_Wavelength_File"]
    CKD_Rad_File         = L01B_Parameters["CKD_Rad_File"  ] 
    CKD_DoLP_File         = L01B_Parameters["CKD_DoLP_File"  ] 
    CKD_Sol_File         = L01B_Parameters["CKD_Sol_File"  ] 
    # Output is stored at
    L1B_File              = L01B_Parameters["L1B_File"     ]
    

    # Print some parameters from the input file 
    print("Parameter settings (subset):")
    print("Number of Viewports    : {0:3d}".format(NViewports))
    print("Viewports              : ",Viewports)
    #print("Number of Swath angles : {0:3d}".format(NSwaths))
    print("# of wavelengths (DoLP): {0:3d}".format(NWavelengths_dolp))
    print("Wavelengths            :  ",Wavelengths_dolp)
    print("CKD_LOS_File           :",CKD_LOS_File)
    print("CKD_Wavelength_File    :",CKD_Wavelength_File)
    print("CKD_Rad_File           :",CKD_Rad_File)
    print("CKD_DoLP_File           :",CKD_DoLP_File)
    print("Output LB1 File        :",L1B_File)


    # imagefiles and nasdatfiles are lists of ALL the datafiles of the entire flight
    imagefiles  = getimagefiles(flightdirectory)
    nasdatfiles = getnasdatfiles(flightdirectory)
    imagefiles.sort()
    nasdatfiles.sort()
    print("imagefiles:",imagefiles)
    print("nasdatfiles:",nasdatfiles)
        
    # collect some general data applicable to all the data files of the flight
    # and needed to make the proper extraction from the CKD data -  in particular the Region Of Interest (ROI)
    for imgfile in imagefiles:
        try:
            f = h5py.File(flightdirectory+imgfile, 'r')
            break  # as soon as the read was successful, exit the loop
        except:
            print("skipping file",imgfile)
    print("\n\nOpening flight data file\n",flightdirectory+imgfile ) 
    
    fattrs = f.attrs
    # fkeys  = f.keys()
    # print file header information
    # print the root attributes content
    #print("Root attributes:")
    #for att in fattrs:
    #    print(" Attribute {0:16s} value : {1:8d}".format(att,fattrs[att]))
        
    #Number of heartbeats in the file
    YEAR         = fattrs['YEAR']
    MONTH        = fattrs['MONTH']
    DAY          = fattrs['DAY']
    Y_DIM        = fattrs["Y_DIM"]
    X_DIM        = fattrs["X_DIM"]
    Y_START      = fattrs['Y_START']
    X_START      = fattrs['X_START']
    AVERAGES     = fattrs['AVERAGES']
    FPU_FRAME_TIME = fattrs['FPU_FRAME_TIME']

    Nrow      = Y_DIM
    Ncol      = X_DIM
    rowoffset = Y_START
    NAverages = AVERAGES

    # overrule X_START  - Offset incorrectly set in camera for any value not a multiple of 4
    if X_START == 277:   
        coloffset = 276   
    else:
        coloffset = X_START

    imagedata = f["IMAGES"] 
    # extract exposure time in [s] from the file
    exposuretime = imagedata[0]['CAM_EXPOSURE'] *1.0E-6   # maybe read it each time -might not be constant

    nbitshift = Setnbitshift(NAverages)
 
    # Store the Region of Interest of the CCD in a dictionary" 
    roi = {'nrow':Nrow,'ncol': Ncol,'rowoff':rowoffset,'coloff':coloffset}

    print("Data file information:")
    print("f.filename             : {0}".format(f.filename))
    print("Number of rows         : {0:3d}".format(Nrow))
    print("Number of columns      : {0:3d}".format(Ncol))
    print("Row    offset          : {0:3d}".format(rowoffset))
    print("Column offset          : {0:3d}".format(coloffset))
    print("Exposure time:",exposuretime)      
    
    f.close()    
    
    # get start and end-times of thje selected track
    H0 = L01B_Parameters["Tstart"][0]
    M0 = L01B_Parameters["Tstart"][1] 
    S0 = L01B_Parameters["Tstart"][2] 
    H1 = L01B_Parameters["Tend"][0]
    M1 = L01B_Parameters["Tend"][1] 
    S1 = L01B_Parameters["Tend"][2]


    # get subset of files that match the time constraints
    # the process is very slow. Not sure why
    files_img, ihbs_img, nhb_img = FH.findihbs(flightdirectory,imagefiles,H0,M0,S0,H1,M1,S1)
    nfiles_img = len(files_img)
    print("Image files to process         :")
    print(files_img)
    print("Number of heartbeats to process: ",nhb_img) 
    print("ihbs_img:",ihbs_img)
    
    
    if (flightdate=="20160708") or (flightdate=="20160707"): 
        print("ALARM: JULY FLIGHT: UTC FROM NASDAT CORRECTED BY 2 HOURS. TURN THIS OFF IF NOT APPLICABLE!!!")
        print("ALARM: JULY FLIGHT: UTC FROM NASDAT CORRECTED BY 2 HOURS. TURN THIS OFF IF NOT APPLICABLE!!!")
        print("ALARM: JULY FLIGHT: UTC FROM NASDAT CORRECTED BY 2 HOURS. TURN THIS OFF IF NOT APPLICABLE!!!")
        HR_NASDAT_OFFSET = 2    
    else:
        HR_NASDAT_OFFSET = 0
        
    # extend the total time interval of the NASDAT a bit to be certain that a nasdat datum can be found
    # for the image datium    
    H0nd,M0nd,S0nd = mt.addseconds(H0,M0,S0,-30.0)    
    H1nd,M1nd,S1nd = mt.addseconds(H1,M1,S1,+30.0)   
    # check which subset of nasdatfiles match the time selection and how many data
    # should be read from each file
    files_nas,ihbs_nas,nhb_nas = FH.findihbs(flightdirectory,nasdatfiles,H0nd,M0nd,S0nd,H1nd,M1nd,S1nd)
    print("NASDAT files to process:")
    print(files_nas)
    print("number of heartbeats in the nasdat files: ",nhb_nas) 

    
    # Collect the data from the NASDAT files
    # this reads all the data from the nasdat files which are in the track
    hr_nd,min_nd, sec_nd, usec_nd, lat, long, gps_alt, wgs84_alt, true_hdg,\
           track, drift, pitch, roll, sza, sun_az_grd =\
           readNASDATfiles(flightdirectory,files_nas,ihbs_nas)
    print("len(hr_nd) :",len(hr_nd))       
                      
    # subract a possible hour correction from NASDAT and make a single hour-array
    hr_nd = hr_nd - HR_NASDAT_OFFSET
    time_nd = np.array(hr_nd + (min_nd + (sec_nd + usec_nd*1.e-6 )/60.0 )/60.0, dtype=float)
    yaw = -drift 
           

    #### Collect CKD
    # collect LOS CKD
    print("Reading CKD LOS data")
    los_sm, los_sp, wpol_sp, wpol_sm, swathangles = CKD_read_los(CKD_LOS_File,roi)
    khalfwidth = 1   
    irow = np.arange(rowoffset,rowoffset+Nrow,dtype=int)

    NSwaths = los_sm.shape[1]
    print("Number of Swath elements: ", NSwaths)
    print("shape los_sm ",              los_sm.shape)
    print("shape los_sm ",              los_sp.shape)

    # Generate masks for alternative summation
    # this is actually a step that could be done as a CKD
    # make a CKD generating module for it. It will make startup of the code faster. 
    if True: 
        print("\n\nGENERATING WEIGHTMASK")
        masksm, masksp, minlossm, minlossp = make_los_masks(roi,los_sm,los_sp,khalfwidth)
        print("Done making masks\n\n")    

    
    # Collect the line-of-sight  relative to the ER-2
    vpangles, swathvectors = CKD_Make_FOV_Vectors(swathangles,Viewports)    
    print("Viewport angles:",vpangles)
    print("Swath angles:",swathangles)
    print("Swathvector[6,3]:",swathvectors[6,2,:])
    
    # Collect wavelength annotation CKD
    print("Reading CKD Wavelength data")
    w_sm, w_sp, p_sm, p_sp  = CKD_read_wcoefs(CKD_Wavelength_File,roi)
    
    print("shape w_sm ",w_sm.shape)
    print("shape w_sp ",w_sp.shape)
    print("shape p_sm ",p_sm.shape)
    print("shape p_sp ",p_sp.shape)

    #### Read radiometric calibration CKD
    rc_sm,rc_sp = CKD_read_radcal(CKD_Rad_File,roi)

    #### Read Modulation depth and retardance
    retardance_eff,modulationdepth_eff = CKD_read_DoLPcal(CKD_DoLP_File,roi)
    
    # collect Dark with flag = 0 or 1 ; 0 indicates no bitshift and Naverage corrections
    dflag = 0
    CKD_darkimage               = CKD_Read_darkimage(CKD_Dark_File,dflag)    
                
    debugtest = False

    # collect solar spectrum
    solnm,solflux = CKD_read_solarspectrum(CKD_Sol_File)
    # extract solar flux at wavelengths (and do averaging)\
    dolpresolution = 3.0 * np.ones(NWavelengths_dolp)
    radresolution  = 3.0 * np.ones(NWavelengths_radi)
    solfluxatw = sol_flux_at_wavelengths(solnm,solflux,Wavelengths_radi,radresolution)

    if straylight_flag:
        print("Importing darkzonefile")
        darkzonefile = "/Users/martijns/ownCloud/Temp/darkzonefile.txt"
        darkzones = importdarkzones(darkzonefile,roi)


    ###### END COLLECTING CKD 
    
    ###### Start processing flight data
    # Read time form image files
    hr_img, min_img, sec_img, usec_img, hb_img= readTimeFromIMAGEfiles(flightdirectory,files_img,ihbs_img)
    
    ### subtract 0.5*1.75 seconds to center time at halfway of the frame
    # modified tp subtract half the effective exposure time depending on ther number of averages
    half_frame_time =   -(AVERAGES * FPU_FRAME_TIME) //2 
    hr_img,min_img,sec_img,usec_img = mt.addmicroseconds(hr_img, min_img, sec_img,usec_img,half_frame_time)
    time_img  = (np.asarray(hr_img) +  np.asarray(min_img)/60.0 + (np.asarray(sec_img) + np.asarray(usec_img)*1E-6 )/3600.0)
   
    # OVERRULE total number of Heartbeats
    NHB = nhb_img
    # initialize storage arrays
    
    dolp             =  np.zeros([NHB,NViewports,NSwaths,NWavelengths_dolp],dtype=float)
    aolp             =  np.zeros([NHB,NViewports,NSwaths,NWavelengths_dolp],dtype=float)
    radi             =  np.zeros([NHB,NViewports,NSwaths,NWavelengths_radi],dtype=float)
    q                =  np.zeros([NHB,NViewports,NSwaths,NWavelengths_dolp],dtype=float)
    u                =  np.zeros([NHB,NViewports,NSwaths,NWavelengths_dolp],dtype=float)
    s_hires          =  np.float32(np.zeros([NHB,NViewports,NSwaths,Nrow],dtype=float)) 
    p_hires          =  np.float32(np.zeros([NHB,NViewports,NSwaths,Nrow],dtype=float))  
    rad_hires        =  np.float32(np.zeros([NHB,NViewports,NSwaths,Nrow],dtype=float))  
    # swathvectors_earth=  np.zeros([NHB,NViewports,NSwaths,3           ],dtype=float) #redundant, 
    # NASDAT arrays
    lat_out           =  np.zeros(NHB, dtype=float)
    lon_out           =  np.zeros(NHB, dtype=float)
    gps_alt_out       =  np.zeros(NHB, dtype=float)
    track_out         =  np.zeros(NHB, dtype=float)
    true_hdg_out      =  np.zeros(NHB, dtype=float)
    pitch_out         =  np.zeros(NHB, dtype=float)
    roll_out          =  np.zeros(NHB, dtype=float)
    yaw_out           =  np.zeros(NHB, dtype=float)
    sza_out           =  np.zeros(NHB, dtype=float)
    saz_out           =  np.zeros(NHB, dtype=float)

    heartbeat         =  np.zeros(NHB, dtype=int)
    hour              =  np.zeros(NHB, dtype=int)
    minute            =  np.zeros(NHB, dtype=int)
    second            =  np.zeros(NHB, dtype=int)
    microsecond       =  np.zeros(NHB, dtype=int)

    dolp_hb = 1.0
    aolp_hb = 0.0
    radi_hb = 1.0 
   
    #print("len(time_nd)",len(time_nd)) 
    #print("time_nd[0]:{0}, time_nd[last]:{1}".format(time_nd[0],time_nd[len(time_nd)-1]))
    heartbeat   = hb_img
    hour        = hr_img
    minute      = min_img
    second      = sec_img 
    microsecond = usec_img
    ii = 0 
    print(">> INTERPOLATING THE NASDAT DATA")   
    for hb in range(0,nhb_img):
        #time_hb = (hour[ii] + minute[ii]/60.0 + (second[ii] +\
        #          microsecond[ii]*1E-6 )/3600.0   ).astype(float)
        time_hb = time_img[ii]
        #print("hr,min,sec",hour[ii],minute[ii],second[ii])
        #print("heartbeat time: {0}".format(time_hb))

        indx = find_nearest(time_nd,time_hb)
        tnd1 = time_nd[indx]
        tnd2 = time_nd[indx+1]
        
        #print("indx,tnd1,tnd2,",indx,tnd1,tnd2)
        lat_out     [ii]  = linterpolate2p(tnd1,tnd2, lat[indx],         lat[indx+1],        time_hb) 
        lon_out     [ii]  = linterpolate2p(tnd1,tnd2, long[indx],       long[indx+1],        time_hb) 
        gps_alt_out [ii]  = linterpolate2p(tnd1,tnd2, gps_alt[indx], gps_alt[indx+1],        time_hb)
        track_out   [ii]  = linterpolate2p(tnd1,tnd2, track[indx],     track[indx+1],        time_hb) 
        true_hdg_out[ii]  = linterpolate2p(tnd1,tnd2, true_hdg[indx], true_hdg[indx+1],      time_hb)         
        pitch_out   [ii]  = linterpolate2p(tnd1,tnd2, pitch[indx],     pitch[indx+1],        time_hb) 
        roll_out    [ii]  = linterpolate2p(tnd1,tnd2, roll[indx],       roll[indx+1],        time_hb) 
        yaw_out     [ii]  = linterpolate2p(tnd1,tnd2, yaw[indx],         yaw[indx+1],        time_hb)  
        sza_out     [ii]  = linterpolate2p(tnd1,tnd2, sza[indx],         sza[indx+1],        time_hb)
        saz_out     [ii]  = linterpolate2p(tnd1,tnd2, sun_az_grd[indx], sun_az_grd[indx+1],  time_hb)

        # print("lat[indx],lat[indx+1],lat_out", lat[indx],lat[indx+1], lat_out[ii])         
        # determine swath angles in Earth frame
        # present approach: perfect pointing -no roll, pitch, or yaw
        ii += 1
    print(">> DONE INTERPOLATING THE NASDAT DATA")  
 
    ######  MAIN LOOP OVER VIEWPORTS, SWATH ANGLES AND WAVELENGTHS 
    print("MAIN LOOP")
    print("Processing nfiles: ",nfiles_img)
    # RESET COUNTER
    ii = 0
    for ifile in range(0,nfiles_img):
        imagefile = files_img[ifile]
        print("ifile, imagefile",ifile,imagefile)
        fimg = h5py.File(flightdirectory+imagefile,'r')
        imagedata  = fimg["IMAGES"]
        locatedata = fimg["LOCATE"] 
        housekdata = fimg["HOUSEK"]
        accelrdata = fimg["ACCELR"]
        fattrs = fimg.attrs
        NAverages    = fattrs["AVERAGES"]

        hb1 = ihbs_img[ifile,0]
        hb2 = ihbs_img[ifile,1]  

        for hb in range(hb1,hb2+1):
            time_hb = time_img[ii]
            #print("\ncollecting image")
            hbii         = imagedata[hb]["HEART_BEAT"]   
            speximage    = imagedata[hb]["IMAGE"].astype(float)
            exposuretime = imagedata[hb]["CAM_EXPOSURE"]*1.0E-6
            #print("done collecting image")
            gpsstring = str(locatedata["GPS"][hb].tostring())
            # overrule the lon,lat from NASDAT by lon,lat from SPEX-airborne GPS
            timeL,hrL,minL,secL, lonL, latL, hL = decodeGPSword(gpsstring)
            # in case that NASDAT TIME IS WRONG (05-07-2016), the GPS time is leading. NASDAT TIME IS THEN NOT USED
            if(useGPSlonlat):
                lonGPS = np.mean(lonL)
                latGPS = np.mean(latL)
            else:    
                lonGPS = mt.resample(timeL,lonL,time_hb)
                latGPS = mt.resample(timeL,latL,time_hb)
            lon_out[ii] = lonGPS
            lat_out[ii] = latGPS

             
            if debugtest:
                speximage = speximage - np.min(speximage)
            else:
                speximage = CorrectDarkcurrent(speximage,CKD_darkimage)

            if straylight_flag:
                #print(">>>>>>> TEST: CORRECTING FOR STRAYLIGHT")
                # experimental procedure to correct straylight 
                speximage = CorrectStraylight(speximage,darkzones)
            speximage = BitshiftCorrection(speximage,nbitshift)
            speximage = PerUnitTime(speximage,NAverages,exposuretime)
            
            # extractLOS using masks is default.  
            #spectra_sm, spectra_sp = extractlos(speximage,los_sm,los_sp, weights_sm, weights_sp,istart_sm, istart_sp)
            #print("Extracting Spectra\n")
            spectra_sm, spectra_sp = extractlos_using_mask(speximage,masksm,masksp,minlossm,minlossp)            
            for ivp in range(0,NViewports):
                print("Frame:{0:3d}, Heartbeat:{1:3d}, Time: {2:2d}:{3:2d}:{4:2d}, VP:{5:1d}".format(hb,\
                      hbii,hour[ii],minute[ii],second[ii],Viewports[ivp]),end="\r")
                for jswath in range(0,NSwaths):
                    #print("Frame:{0:3d}, Heartbeat:{1:3d}, Time: {2:2d}:{3:2d}:{4:2d},\
                    #          VP:{5:1d}, Swath:{6:2d}".format(hb,hbii,hour[ii],minute[ii],\
                    #          second[ii],Viewports[ivp],jswath),end="\r")
                    #kmaxsmin        = los_sm                [ivp,jswath,:]
                    #kmaxsplus       = los_sp                [ivp,jswath,:]
                    # Retardance is still used to determine the length of the length of the fit-interval. 
                    retardance      = retardance_eff        [ivp,jswath,:]
                    ModAmpW         = modulationdepth_eff   [ivp,jswath,:]

                    Radcalcoef_sm   = rc_sm  [ivp,jswath,:]    
                    Radcalcoef_sp   = rc_sp  [ivp,jswath,:]
                       
                    wavel_s         = w_sm                  [ivp,jswath,:] 
                    wavel_p         = w_sp                  [ivp,jswath,:]
                    #psm             = p_sm                  [ivp,jswath,::-1]
                    #psp             = p_sp                  [ivp,jswath,::-1]
                    
                    # I do not know why I chose to override the above wavelengths with wavelengts
                    # recalculated from the polynomial. 
                    # wavel_s,wavel_p  = Assign_Wavelengths(psm,psp,roi) 
                    #LOSPixelSelection is commented out with new method
                    #[s,p] = LOSPixelSelection(speximage,irow,kmaxsplus,kmaxsmin,khalfwidth)

                    s = spectra_sm[ivp,jswath,:]
                    p = spectra_sp[ivp,jswath,:]

                    # Align S+ with S- spectrum - resampled at S- wavelength
                    # should also work with resampling, apparently I chose not to. Check why
                    # p_algn      =    mt.resample(wavel_p,p,wavel_s)
                    # p_algn      =    AlignSpwithSm(p,psm,psp,roi)
                    p_algn      =    mt.interpolate(wavel_p,p,wavel_s,'cubic')

                    s_rad = s      * Radcalcoef_sm
                    p_rad = p_algn * Radcalcoef_sp

                    radiance    = 0.5*(s_rad + p_rad) 
                    diffsp      = 0.5*(s_rad - p_rad)

                    s_hires  [ii,ivp,jswath,:] = s_rad
                    p_hires  [ii,ivp,jswath,:] = p_rad
                    rad_hires[ii,ivp,jswath,:] = radiance
                    
                    # Construct modulation depth
                    modulation  = diffsp / radiance      # make a function for this
                    if not rad_only_flag:
                        for wl in range(0,NWavelengths_dolp):
                            """
                            print("Frame:{0:3d}, Heartbeat:{1:3d}, Time: {2:2d}:{3:2d}:{4:2d},\
                                  VP:{5:1d}, Swath:{6:2d}, wavelength{7:3d}".format(hb,hbii,hour[ii],minute[ii],\
                                  second[ii],Viewports[ivp],jswath,wl),end="\r")
                            """
                            wavel_select       = Wavelengths_dolp[wl]
                            dolp_hb,aolp_hb,q0,u0,pdemod     = Demodulate(wavel_s,modulation,retardance,ModAmpW,wavel_select)         
                            
                            # TEST DEBUG! put outside the loop
                            # radi_hb = linterpolate(wavel_s,radiance,wavel_select)
                            
                            #radi_hb = radi_hb / ((1+0.5*(msm12[wrng]+msp12[wrng])*qnl + 0.5*(msm13[wrng]+msp13[wrng])*unl))
                            #txt =  "{0:8d}, {1:8e}, {2:8e}, {3:8e}, {4:8e}, {5:8e}, {5:8e} \n".format(\
                            #heartbeat[ii],dolp[ii],aolp,rad_wavel_select[ii], A[0],B[0],C[0])
                            q_hb=q0
                            u_hb=u0
                            dolp[ii,ivp,jswath,wl] = dolp_hb
                            aolp[ii,ivp,jswath,wl] = aolp_hb 
                            q   [ii,ivp,jswath,wl] = q_hb
                            u   [ii,ivp,jswath,wl] = u_hb                        

                    # end wavelength loop 
                    radi[ii,ivp,jswath,:] = mt.interpolate(wavel_s,radiance,Wavelengths_radi,'cubic')
                # end swath loop
            # end viewport loop
            ii+=1
        # end heartbeat loop 
        fimg.close()
    # end image file loop
    print("Total Number of HeartBeats processes:",ii)
    print("Reserved Number of HeartBeats       :",NHB)
    hblast = ii-1
  


    
    # Generate arrays for consistency with Jeroen's L1B data products
    # adopt 3nm
    # smooth solar spectrum 
    
    dolpresolution = 3.0 * np.ones(NWavelengths_dolp)
    radresolution  = 3.0 * np.ones(NWavelengths_radi)
    
    # do the smooting of solar spectrum
    sol_flux = sol_flux_at_wavelengths(solnm,solflux,Wavelengths_radi,radresolution)
    
    # do a quick-dirty geolocation (no terrain, no ellipsoid, flat earth, gps height is THE height)
    
    # DEBUG: replace trueheading with track for the Falcon, ignoring possible error in yaw compensation
    if True:
        wheretolook = track_out
        #rotate all swathvectors 180 degrees because youre in the falcon
        swathvectors[:,:,0] = - swathvectors[:,:,0]
        swathvectors[:,:,1] = - swathvectors[:,:,1]        
    else:
        wheretolook = true_hdg_out
    
    lonswath, latswath = geolocateviewvectors(swathvectors,lon_out,lat_out,roll_out,pitch_out,\
                                              wheretolook,gps_alt_out)
    # Store the Level 1B data product

    while(os.path.isfile(L1B_File)):
        print("Output file already exists, renaming")
        L1B_File = L1B_File.split(".")[0]+"new.h5"
    try:
        fh5out = h5py.File(L1B_File, 'w')
        print("successfully opened file",L1B_File)
    except IOError:
         print("error opening output file", L1B_File)
         exit(1)
    
    
    # construct 2D array for VIEWPORTS
    vpout = np.zeros([NViewports,2],dtype='float')
    vpout[:,0] = Viewports[:]
    vpout[:,1] = vpangles[:]    
    # Make a group called "INFO"      
    ginfoout = fh5out.create_group("INFO")
    ginfoout.attrs["HEARTBEATS"]     =   NHB
    ginfoout.attrs["IMG_MIN"]        =   heartbeat[0]
    ginfoout.attrs["IMG_MAX"]        =   heartbeat[hblast]
    ginfoout.attrs["X_DIM"]          =   X_DIM 
    ginfoout.attrs["Y_DIM"]          =   Y_DIM 
    ginfoout.attrs["X_START"]        =   X_START
    ginfoout.attrs["Y_START"]        =   Y_START
    ginfoout.attrs["AVERAGES"]       =   AVERAGES    
    ginfoout.attrs["NVIEWPORTS"]     =   NViewports
    ginfoout.attrs["NSWATHS"]        =   NSwaths
    ginfoout.attrs["NWAVELENGTHS"]   =   NWavelengths_dolp
    ginfoout.attrs["FPU_FRAME_TIME"] =   FPU_FRAME_TIME 
    ginfoout.attrs["YEAR"]           =   YEAR
    ginfoout.attrs["MONTH"]          =   MONTH      
    ginfoout.attrs["DAY"]            =   DAY        
    ginfoout.create_dataset("VIEWPORTS",       data = vpout )
    ginfoout.create_dataset("SWATHANGLES",     data = swathangles)
    ginfoout.create_dataset("SWATHVECTORS",    data = swathvectors) 
    ginfoout.create_dataset("WAVELENGTHS",     data = Wavelengths_dolp) 
    ginfoout.create_dataset("WAVELENGTHS_RAD", data = Wavelengths_radi)     
    ginfoout.create_dataset("DOLPRESOLUTION",  data = dolpresolution)
    ginfoout.create_dataset("RADRESOLUTION",   data = radresolution)
    ginfoout.create_dataset("SOLARSPECTRUM",   data = solfluxatw)
 
   
    # Dump the level 1B data 
    level1b                = fh5out.create_group("LEVEL1B")
    l1bdolp                = level1b.create_dataset("DOLP",    data=dolp)
    l1baolp                = level1b.create_dataset("AOLP",    data=aolp)
    l1brad                 = level1b.create_dataset("RADIANCE",data=radi)
    # make RMSDIFF dataset for compatibility with L1B format spec
    level1b.create_dataset("RMSDIFF",data=np.ones(dolp.shape))


    if(dumphires):
        spectragrp = fh5out.create_group("SPECTRAHIRES")
        spectragrp.create_dataset("WAVELENGTHS-S,",data=w_sm)
        spectragrp.create_dataset("WAVELENGTHS-P,",data=w_sp)
        spectragrp.create_dataset("RADIANCE", data=rad_hires)
        spectragrp.create_dataset("S", data=s_hires)
        spectragrp.create_dataset("P", data=p_hires)


    # dataset below may be deleted - it is a placeholder for SWATHVECTORS relative to Earth. 
    # But instead there will be a dataset containing the long-lat for each SWATH Element
    # level1b.create_dataset("SWATHVECTORS_EARTH", data = swathvectors_earth )  

    l1bdolp.attrs["UNITS"]  = "[]"
    l1brad.attrs ["UNITS"]  = "Watt/m2/nm/solid angle"
    l1baolp.attrs["UNITS"]  = "Degrees"
  
    l1btime = fh5out.create_group("TIME")
    l1btime.create_dataset("HEART_BEAT",   data = heartbeat)
    l1btime.create_dataset("HOUR",         data = hour)
    l1btime.create_dataset("MINUTE",       data = minute)
    l1btime.create_dataset("SECOND",       data = second)
    l1btime.create_dataset("MICROSECOND",  data = microsecond)
        
    # dump the NASDAT data
    nasdat = fh5out.create_group("NASDAT")
    nasdat.create_dataset("LAT",          data = lat_out)
    nasdat.create_dataset("LON",          data = lon_out) 
    nasdat.create_dataset("GPS_ALT",      data = gps_alt_out)
    nasdat.create_dataset("TRACK",        data = track_out)
    nasdat.create_dataset("HEADING",      data = true_hdg_out)
    nasdat.create_dataset("PITCH",        data = pitch_out)
    nasdat.create_dataset("ROLL",         data = roll_out)
    nasdat.create_dataset("YAW",          data = yaw_out)
    nasdat.create_dataset("SOLAR_ZENITH", data = sza_out)
    nasdat.create_dataset("SOLAR_AZ_GRD", data= saz_out)       

    # dump the geolocationdata
    if True :    
        geoloc = fh5out.create_group("GEOLOCATION")
        geoloc.create_dataset("LONGITUDE",data=lonswath)
        geoloc.create_dataset("LATITUDE",data=latswath)
    # Close open files 

    fh5out.close()


    ### END OF LEVEL 0-1B DATAPROCESSOR


def main(argv):

    if len(argv) == 0:
        print ("")
        print ("\tNo input argument provided")
        sys.exit(1)

    inputfile = argv[0]

    if not os.path.isfile(inputfile):
        print ("")
        print ("\tFile {0:s} doesn't exists".format(inputfile))
        sys.exit(1)


    L01B_Issue2p2(inputfile)

if __name__ == '__main__':
    main(sys.argv[1:])

