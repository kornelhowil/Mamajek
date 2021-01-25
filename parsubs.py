#parsubs getpsi and geta routines rewritten in python by KR

import numpy as np

def getpsi(phi, ecc):
	pi = 3.14159265
	psi= phi
	for i in [1,2,3,4]:
		fun = psi - ecc*np.sin(psi)
		dif = phi - fun
		der = 1 - ecc*np.cos(psi)
		psi = psi + dif/der
	return psi

def cross(a,b):
	c1 = a[1]*b[2] - a[2]*b[1]
	c2 = a[2]*b[0] - a[0]*b[2]
	c3 = a[0]*b[1] - a[1]*b[0]

	return np.array([c1,c2,c3])

def dot(a,b):
	return a[0]*b[0] + a[1] * b[1] + a[2]*b[2] 

def geta(hjd, alpha, delta, t0par):
	spring = np.array([1.,0.,0.])
	summer = np.array([0.,0.9174,0.3971])

	north = np.array([0.,0.,1.])

	pi = 3.14159265
	ecc = 0.0167
	vernal = 2719.55
	offset = 75
	peri   = vernal - offset
	phi = (1 - offset/365.25)*2*pi

	psi = getpsi(phi, ecc)
	costh = (np.cos(psi) - ecc)/(1-ecc*np.cos(psi))	
	sinth = -np.sqrt(1-costh**2)

	xpos = np.array([1.,1.,1.])
	ypos = np.array([1.,1.,1.])

	for i in [0,1,2]:
		xpos[i] = spring[i]*costh + summer[i]*sinth
		ypos[i] =-spring[i]*sinth + summer[i]*costh

	radian = 180/3.14159265
	rad = np.array([1.,1.,1.])

	rad[0] = np.cos(alpha/radian)*np.cos(delta/radian)
	rad[1] = np.sin(alpha/radian)*np.cos(delta/radian)
	rad[2] = np.sin(delta/radian)

	east = cross(north, rad)
	e2 = dot(east,east)
 
	east = east/np.sqrt(e2)

	north = np.cross(rad, east)
	phi   = (t0par+1 - peri)/365.25*2*pi

	psi = getpsi(phi, ecc)
	
	qn2 = 0
	qe2 = 0

	sun = np.array([1.,1.,1.])

	for i in [0,1,2]:
		sun[i] = xpos[i]*(np.cos(psi)-ecc) + ypos[i]*np.sin(psi)*np.sqrt(1-ecc**2)
		qn2 = qn2 + sun[i]*north[i]
		qe2 = qe2 + sun[i]*east[i]

	phi   = (t0par - 1. - peri)/365.25*2*pi
	psi = getpsi(phi, ecc)

	qn1 = 0
	qe1 = 0
	for i in [0,1,2]:
		sun[i] = xpos[i]*(np.cos(psi)-ecc) + ypos[i]*np.sin(psi)*np.sqrt(1-ecc**2)
		qn1 = qn1 + sun[i]*north[i]
		qe1 = qe1 + sun[i]*east[i]

	phi   = (t0par - peri)/365.25*2*pi
	psi = getpsi(phi, ecc)
	qn0 = 0
	qe0 = 0
	for i in [0,1,2]:
		sun[i] = xpos[i]*(np.cos(psi)-ecc) + ypos[i]*np.sin(psi)*np.sqrt(1-ecc**2)
		qn0 = qn0 + sun[i]*north[i]
		qe0 = qe0 + sun[i]*east[i]

	vn0 = (qn2-qn1)/2.
	ve0 = (qe2-qe1)/2.

	factor = 365.25*4.74
	t = hjd
	phi   = (t - peri)/365.25*2*pi

	psi = getpsi(phi, ecc)
	qn = -qn0 - vn0*(t-t0par)
	qe = -qe0 - ve0*(t-t0par)
	#qn = 0
	#qe = 0

	for i in [0,1,2]:
		sun[i] = xpos[i]*(np.cos(psi)-ecc) + ypos[i]*np.sin(psi)*np.sqrt(1-ecc**2)
		qn = qn + sun[i]*north[i]
		qe = qe + sun[i]*east[i]

	return qn, qe

def get_sun_v(alpha, delta, t0par):
	spring = np.array([1.,0.,0.])
	summer = np.array([0.,0.9174,0.3971])

	north = np.array([0.,0.,1.])

	pi = 3.14159265
	ecc = 0.0167
	vernal = 2719.55
	offset = 75
	peri   = vernal - offset
	phi = (1 - offset/365.25)*2*pi

	psi = getpsi(phi, ecc)
	costh = (np.cos(psi) - ecc)/(1-ecc*np.cos(psi))	
	sinth = -np.sqrt(1-costh**2)

	xpos = np.array([1.,1.,1.])
	ypos = np.array([1.,1.,1.])

	for i in [0,1,2]:
		xpos[i] = spring[i]*costh + summer[i]*sinth
		ypos[i] =-spring[i]*sinth + summer[i]*costh

	radian = 180/3.14159265
	rad = np.array([1.,1.,1.])

	rad[0] = np.cos(alpha/radian)*np.cos(delta/radian)
	rad[1] = np.sin(alpha/radian)*np.cos(delta/radian)
	rad[2] = np.sin(delta/radian)

	east = cross(north, rad)
	e2 = dot(east,east)
 
	east = east/np.sqrt(e2)

	north = np.cross(rad, east)
	phi   = (t0par+1 - peri)/365.25*2*pi

	psi = getpsi(phi, ecc)
	
	qn2 = 0
	qe2 = 0

	sun = np.array([1.,1.,1.])

	for i in [0,1,2]:
		sun[i] = xpos[i]*(np.cos(psi)-ecc) + ypos[i]*np.sin(psi)*np.sqrt(1-ecc**2)
		qn2 = qn2 + sun[i]*north[i]
		qe2 = qe2 + sun[i]*east[i]

	phi   = (t0par - 1. - peri)/365.25*2*pi
	psi = getpsi(phi, ecc)

	qn1 = 0
	qe1 = 0
	for i in [0,1,2]:
		sun[i] = xpos[i]*(np.cos(psi)-ecc) + ypos[i]*np.sin(psi)*np.sqrt(1-ecc**2)
		qn1 = qn1 + sun[i]*north[i]
		qe1 = qe1 + sun[i]*east[i]

	phi   = (t0par - peri)/365.25*2*pi
	psi = getpsi(phi, ecc)
	qn0 = 0
	qe0 = 0
	for i in [0,1,2]:
		sun[i] = xpos[i]*(np.cos(psi)-ecc) + ypos[i]*np.sin(psi)*np.sqrt(1-ecc**2)
		qn0 = qn0 + sun[i]*north[i]
		qe0 = qe0 + sun[i]*east[i]

	vn0 = (qn2-qn1)/2.
	ve0 = (qe2-qe1)/2.


	return vn0, ve0