'''
      ! this calculates the value of Eq. 18 from V.Batista et al. MOA-2009-BLG-387
      ! but for my particular event OGLE-2009-BLG-020, so thick disk
      !
      ! 1. expected galactic density (thick disk)
      ! 2. likelihood of the given proper motion (for a thick disk lens)
      ! 3. mass function
      ! 3. jacobian  || d(Dl, M, _mu_) / d(tE, _mu_, _piE_) ||
      !
      ! weights of every link should be muliplied by the value of 'get_gal_jac'
      ! to account for priors and trasformation from physical to ulensing parameters
      ! thetaE ----> _mu_
'''

import numpy as np

def get_gal_jac(ra, dec, pien, piee, murel, te, pis, t0par,ve_s_1,ve_s_2,gl,gb,masspower=1.):
      #ra, dec in degrees
      debug=False
      vrot=220.
      dec2ra=3.14159265359/180.

      # lens and source distances
      thetae=murel*(te/365.25)  ######### thetae=murel*te #
      ds = 1./pis
      pie=np.sqrt(pien**2+piee**2)
      pil = pis + thetae*pie
      dl=1./pil
      mass=thetae/(8.14*pie) # in Solar mass
      if(debug): print('dl =',dl ,' ds =',ds)
      if(debug): print('pil=',pil,' pis=',pis)

      # in galactic coordinates, the PA or equatorial north pole, approximate angle
      northPA=60.*dec2ra

      #--------------------------------------------
      # RELATIVE PROPER MOTION

      # Earth velocity at t0par relative to the Sun
      # so as taken from geta routine but with minus sign
      # ve_s_1 north in km/s
      # ve_s_2 east  in km/s
      ve_gs_1=ve_s_1*np.cos(northPA)-ve_s_2*np.sin(northPA) # north galactic coord in km/s
      ve_gs_2=ve_s_1*np.sin(northPA)+ve_s_2*np.cos(northPA) # east  galactic coord in km/s
      ve_g_1= 7.0 + ve_gs_1         # Earth velocity in respect to the Galaxy
      ve_g_2=12.0 + ve_gs_2 + vrot  # Earth velocity in respect to the Galaxy
      if(debug): print('earthv=',ve_g_1,' ',ve_g_2)

      # expected thick disk velocities
      vd_1=0.
      vd_2=vrot-20.  # rotation + asymmetric drift, the stars do not keep up with the rotation
      # expected dispersion in disk velocities
      svd_1=40.
      svd_2=55. # thick disk is a little more unsettled than thin disk
      if(debug): print('expected disk v=',vd_1,'+-',svd_1,',',vd_2,'+-',svd_2)

      # expected bulge source velocities
      vb_1=0.
      vb_2=0.
      # expected dispersion in disk velocities
      svb_1=80.
      svb_2=80. # bulge motions are quite random
      if(debug): print('expected bulge v=',vb_1,'+-',svb_1,',',vb_2,'+-',svb_2)
      
      # relative proper motion we expect
      mu_exp_1=( (vd_1 - ve_g_1)*pil - (vb_1 - ve_g_1)*pis)/4.74  # in AU/yr/kpc = mas/yr 
      mu_exp_2=( (vd_2 - ve_g_2)*pil - (vb_2 - ve_g_2)*pis)/4.74  # in AU/yr/kpc = mas/yr

      # relative proper motion dispersion we expect
      smu_exp_1= np.sqrt( (svd_1*pil)**2 + (svb_1*pis)**2 )/4.74       # in mas/yr
      smu_exp_2= np.sqrt( (svd_2*pil)**2 + (svb_2*pis)**2 )/4.74       # in mas/yr
      if(debug): print('expected mu=',mu_exp_1,'+-',smu_exp_1,',',mu_exp_2,'+-',smu_exp_2)

      # relative proper motion we see
      mu_len=thetae/te*365.25
      mu_l_1=mu_len*(pien*np.cos(northPA)-piee*np.sin(northPA))/pie       # north galactic component in mas/yr
      mu_l_2=mu_len*(pien*np.sin(northPA)+piee*np.cos(northPA))/pie       # east  galactic component in mas/yr
      if(debug): print('observed mu=',mu_l_1,',',mu_l_2)

      # probability of given proper motion
      fmu_1=np.exp( -(mu_exp_1-mu_l_1)**2/(2.*smu_exp_1**2) )/smu_exp_1       # prob
      fmu_2=np.exp( -(mu_exp_2-mu_l_2)**2/(2.*smu_exp_2**2) )/smu_exp_2       #
      if(debug): print('mu prob=',fmu_1,',',fmu_2)
      

      #--------------------------------------------
      # DENSITY
      gl=gl*dec2ra
      gb=gb*dec2ra
      if(debug): print('ra,dec(deg) =',ra,',',dec)
      if(debug): print('l,b(rad) =',gl,',',gb)

      xg = 8. - dl*np.cos(gl)*np.cos(gb)
      yg = dl*np.sin(gl)*np.cos(gb)
      zg = dl*np.sin(gb)
      rg = np.sqrt(xg**2+yg**2)
      if(debug): print('x,y,z =',xg,',',yg,',',zg)
      if(debug): print('r(kpc) =',rg)

      # thick disk scale height and scale lenght
      sh=0.6  # kpc
      sl=2.75 # kpc
      # exponential disk model with scale height = 600pc, scale length=2.75kpc.
      densprob = np.exp(-abs(rg)/sl)*np.exp(-abs(zg)/sh)
      if(debug): print('dens prob =',densprob)
     
      #--------------------------------------------
      # very simple mass function weighting assumed here
      mass_function = 1./(mass**masspower)
 
      #--------------------------------------------
      # TOTAL + JACOBIAN

      gal_jac = 1.e5*densprob*fmu_1*fmu_2*(mass_function*mass)*(dl**4)*((thetae/te)**4)*te/pie # *te ze zmiany zmiennej w jakobianie (thetaE na mu)
      if(debug): print('gal_jac_dens =', gal_jac)
      return gal_jac


#BROKEN!
#computes velocity vector for parameters of the lens provided, some not used, copied from get_gal_jac
#returns lens velocity in N,E galactic coords, also Earth velocity in NEgal
#      return vel_1, vel_2, ve_g_1, ve_g_2

def get_vel(ra, dec, pien, piee, murel, te, pis, t0par,ve_s_1,ve_s_2,gl,gb,masspower=1.):
      #ra, dec in degrees
      debug=False
      vrot=220.
      dec2ra=3.14159265359/180.

      # lens and source distances
      thetae=murel*(te/365.25)  ######### thetae=murel*te #
      ds = 1./pis
      pie=np.sqrt(pien**2+piee**2)
      pil = pis + thetae*pie
      dl=1./pil
      mass=thetae/(8.14*pie) # in Solar mass
      if(debug): print('dl =',dl ,' ds =',ds)
      if(debug): print('pil=',pil,' pis=',pis)

      # in galactic coordinates, the PA or equatorial north pole, approximate angle
      northPA=60.*dec2ra

      #--------------------------------------------
      # RELATIVE PROPER MOTION

      # Earth velocity at t0par relative to the Sun
      # so as taken from geta routine but with minus sign
      # ve_s_1 north in km/s
      # ve_s_2 east  in km/s
      ve_gs_1=ve_s_1*np.cos(northPA)-ve_s_2*np.sin(northPA) # north galactic coord in km/s
      ve_gs_2=ve_s_1*np.sin(northPA)+ve_s_2*np.cos(northPA) # east  galactic coord in km/s
      ve_g_1= 7.0 + ve_gs_1         # Earth velocity in respect to the Galaxy
      ve_g_2=12.0 + ve_gs_2 + vrot  # Earth velocity in respect to the Galaxy
      if(debug): print('earthv=',ve_g_1,' ',ve_g_2)

      # expected thick disk velocities
      vd_1=0.
      vd_2=vrot-20.  # rotation + asymmetric drift, the stars do not keep up with the rotation
      # expected dispersion in disk velocities
      svd_1=40.
      svd_2=55. # thick disk is a little more unsettled than thin disk
      if(debug): print('expected disk v=',vd_1,'+-',svd_1,',',vd_2,'+-',svd_2)

      # expected bulge source velocities
      vb_1=0.
      vb_2=0.
      # expected dispersion in disk velocities
      svb_1=80.
      svb_2=80. # bulge motions are quite random
      if(debug): print('expected bulge v=',vb_1,'+-',svb_1,',',vb_2,'+-',svb_2)
      
      # relative proper motion we expect
      mu_exp_1=( (vd_1 - ve_g_1)*pil - (vb_1 - ve_g_1)*pis)/4.74  # in AU/yr/kpc = mas/yr
      mu_exp_2=( (vd_2 - ve_g_2)*pil - (vb_2 - ve_g_2)*pis)/4.74  # in AU/yr/kpc = mas/yr

      #mul*4.74 = (vd-ve)*pil
      #mul*4.74/pil = vd - ve
      vel_1 = mul_1*4.74/pil# + ve_g_1
      vel_2 = mul_2*4.74/pil# + ve_g_2

      #returns lens velocity in N,E galactic coords, also Earth velocity in NEgal
      #      return vel_1, vel_2, ve_g_1, ve_g_2

      return vel_1, vel_2, ve_g_1, ve_g_2


