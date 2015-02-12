"""
    ..                  __
    ..      ________   __/  ______    ______
    ..    /_    __  \ __/ /   __  \ /   __  \
    ..     /  /  /  /    /  /__/  //  /  /  /
    ..    /  /  /  /    /  ______//  /  /  /
    ..   /  /__/  /    /  /      /  /__/  /
    .. /_________/    /__/       \_______/  DEMO

    Denoising, Deconvolving, and Decomposing Photon Observations DEMO

"""
from __future__ import division
import __main__ as main_demo
from scipy.signal import sepfir2d as sm2
#from nifty import *
import os
import numpy as np
from nifty import switch,about,                                              \
                  space,point_space,rg_space,                                \
                  field,                                                     \
                  response_operator,                                         \
                  ncmap
import d3po
#reload(d3po)


#about.hermitianize.off()
#about.multiprocessing.off()


##-----------------------------------------------------------------------------

class convolution_operator(response_operator):
    """
        Class for D3PO demonstration response operators

        This response involves a convolution with an exposure mask and a
        given kernel. Additionally, a binary mask excluding pixels hosting
        bright point sources can be included.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        kernel : ndarray
            Normalised 1D convolution kernel with finite support.
        exposure : ndarray
            Exposure array mathching the dimensionality of the domain.
        mask : ndarray
            Masking array mathching the dimensionality of the domain.
        den : bool, *optional*
            Whether to consider the arguments as densities or not;
            mandatory for the correct incorporation of volume weights
            (default: False).
        target : space, *optional*
            The space wherein the operator output lives.

        See Also
        --------
        response_operator

        Raises
        ------
        TypeError
            If domain is no NIFTY space.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        kernel : ndarray
            Normalised 1D convolution kernel with finite support.
        exposure : ndarray
            Exposure array mathching the dimensionality of the domain.
        mask : ndarray
            Masking array mathching the dimensionality of the domain.
        identify_io_pixels : bool
            Boolean indicating whether there is an equivalence between data
            and signal space; i.e., pixels are uniquely associated by their
            raveled indices.
        mask_out_pixels : switch
            Switch controlling the usage of the masking array that excludes
            pixels hosting bright point sources.
        sym : bool
            Indicates whether the operator is self-adjoint or not.
        uni : bool
            Indicates whether the operator is unitary or not.
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not.
        den : bool
            Whether to consider the arguments as densities or not.
            Mandatory for the correct incorporation of volume weights.
        target : space
            The space wherein the operator output lives

    """

    def __init__(self,domain,kernel,exposure,mask,den=True,target=None):
        """
            Sets the standard properties for a response operator and
            `kernel`, `exposure`, and `mask`.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            kernel : ndarray
                Normalised 1D convolution kernel with finite support.
            exposure : ndarray
                Exposure array mathching the dimensionality of the domain.
            mask : ndarray
                Masking array mathching the dimensionality of the domain.
            den : bool, *optional*
                Whether to consider the arguments as densities or not;
                mandatory for the correct incorporation of volume weights
                (default: False).
            target : space, *optional*
                The space wherein the operator output lives.

        """
        ## check domain
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        ## set convolution specifications
        self.kernel = kernel
        self.exposure = self.domain.enforce_values(exposure,extend=False)
        self.mask = self.domain.enforce_values(mask,extend=False)

        ## set response specific attributes
        self.identify_io_pixels = True ## data and signal space are equivalent (i.e., raveled indices are equal)
        self.mask_out_pixels = switch(default=False) ## mask out outshining pixels

        ## set (general) operator attributes
        self.sym = False
        self.uni = False
        self.imp = False
        self.den = bool(den)

        ## set target
        if(target is None):
            target = point_space(self.domain.dim(split=False),datatype=self.domain.datatype)
        self.target = target

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def apply_PSF(self,x):
        """
            Applies a point spread function by convolving the input array with
            the corresponding kernel.

            Parameters
            ----------
            x : ndarray
                Input 2D array.

            See Aslo
            --------
            scipy.signal.sepfir2d

            Returns
            -------
            y : ndarray
                Convolved output 2D array.

        """
        return sm2(x,self.kernel,self.kernel)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the operator to a given field
        ## convolve (forward)
        x_ = self.apply_PSF(x.val)
        x_ *= self.exposure
        if(self.mask_out_pixels.status):
            x_ *= self.mask
        ## return data vector
        return field(self.target,val=x_,target=self.target)

    def _adjoint_multiply(self,x,**kwargs): ## > applies the adjoint operator to a given field
        x_ = self.domain.enforce_shape(x.val)
        ## convolve (backward)
        if(self.mask_out_pixels.status):
            x_ *= self.mask
        x_ *= self.exposure
        x_ = self.apply_PSF(x_)
        ## return signal field
        return field(self.domain,val=x_,target=kwargs.get("target",None))

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

def load_demo(demodirectory):
    """
        Loads all objects relevant for the D3PO demonstration

        Parameters
        ----------
        demodirectory : string
            Directory whereto the demo files are located.

        Raises
        ------
        IOError
            If one of the demo files is missing.

        Retrurns
        --------
        R : convulotion_operator
            Response operaotr of the D3PO demonstration.
        d : field
            Data vector of the D3PO demonstration.

    """
    ## check workspace
    dd = os.path.realpath(demodirectory)
    if(os.path.exists(dd)):
        try:
            ## load arrays
            events = np.genfromtxt(os.path.join(dd,"demo_events.txt"))
            kernel = np.genfromtxt(os.path.join(dd,"demo_kernel.txt"))
            exposure = np.genfromtxt(os.path.join(dd,"demo_exposure.txt"))
            mask = np.genfromtxt(os.path.join(dd,"demo_mask.txt"))
        except(IOError):
            raise IOError(about._errors.cstring("ERROR: demo source files not found in '"+dd+"'."))
    else:
        raise IOError(about._errors.cstring("ERROR: '"+dd+"' nonexisting."))
    ## set response
    x_space = rg_space(events.shape[::-1])
    R = convolution_operator(x_space,kernel,exposure,mask,den=True,target=None)
    ## set data
    d = field(R.target,val=events,target=R.target)
    ## return argument tuple
    return R,d

##-----------------------------------------------------------------------------

##=============================================================================

if(__name__=="__main__"):

    ## set up demo
    demodirectory = os.path.dirname(os.path.realpath(__file__))
    R,d = load_demo(demodirectory)

    ## set up problem
    demo_config = os.path.join(demodirectory,"demo_config")
    problem = d3po.problem(R,demo_config,"./d3po_demo/")

    ## solve problem
    problem.solve(d)

    ## check whether interactive
    if(not hasattr(main_demo,"__file__")):
        ## define nice
        HE = ncmap.he()
        HE.set_bad([0,0,0,1]) ## fix zero event pixels
        nice = {"vmin":1,"vmax":144,"cmap":HE,"norm":"log"}

        ## get results
        s,u,rho_s,rho_u = problem.get(s=True,u=True,rho_s=True,rho_u=True)

        ## nicely plot results
        field(R.domain,val=d.val).plot(title="(raw) photon observation",unit="counts",**nice)
        rho_s.plot(title="diffuse photon flux",unit="counts/pixel",**nice)
        field(R.domain,val=R.apply_PSF(rho_u.val)).plot(title="(reconvolved) point-like flux",unit="counts/pixel",**nice)

        ## plot response properties
        #e = field(R.domain)
        #e[e.dim(split=True)[0]//2,e.dim(split=True)[1]//2] = 10
        #field(R.domain,val=R.apply_PSF(e)).plot(title="convolution kernel",vmin=0,vmax=1,cmap=pl.cm.gray_r)
        #field(R.domain,val=R.exposure).plot(title="exposure mask",vmin=0,vmax=1,cmap=pl.cm.gray)
        #field(R.domain,val=R.mask).plot(title="binary mask",vmin=0,vmax=1,cmap=pl.cm.gray)

        ## plot D123PO
        #l = problem.get(lmbda=True)
        #field(R.domain,val=l.val).plot(title="(reproduced) noiseless observation",unit="counts",**nice)
        #field(R.domain,val=R.apply_PSF(rho_s.val+rho_u.val)).plot(title="(reconvolved) total photon flux",unit="counts/pixel",**nice)
        #(rho_s+rho_u).plot(title="total photon flux",unit="counts/pixel",**nice)

        ## plot signal fields
        #s.plot(title="diffuse signal field",vmin=9.5,vmax=13.7,unit="dimensionless")
        #s.plot(title="diffuse power spectrum",vmin=2E-13,vmax=1,power=True,mono=False,other=problem.S.get_power())
        #u.plot(title="point-like signal field",vmin=13.2,vmax=17.4,unit="dimensionless")

        ## flux uncertainties
        #delta_rho_s,delta_rho_u = problem.get(s_err=True,u_err=True)

##=============================================================================

