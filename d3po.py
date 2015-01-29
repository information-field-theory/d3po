## D3PO (Deonoiseing, Deconvolving, and Decomposing Photon Observations) has
## been developed at the Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2014 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/d3po/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
    ..                  __
    ..      ________   __/  ______    ______
    ..    /_    __  \ __/ /   __  \ /   __  \
    ..     /  /  /  /    /  /__/  //  /  /  /
    ..    /  /  /  /    /  ______//  /  /  /
    ..   /  /__/  /    /  /      /  /__/  /
    .. /_________/    /__/       \_______/

    Denoising, Deconvolving, and Decomposing Photon Observations

    TODO: documentation

"""
from __future__ import division
import imp
import shutil
#from nifty import *
import os
import numpy as np
from multiprocessing import Value as mv
from nifty import switch,notification,about,                                 \
                  field,                                                     \
                  sqrt,                                                      \
                  operator,diagonal_operator,power_operator,                 \
                  infer_power,                                               \
                  invertible_operator,propagator_operator,                   \
                  steepest_descent


#about.hermitianize.off()
#about.multiprocessing.off()


__version__ = "1.0.0"


##-----------------------------------------------------------------------------

def exp(x): ## overwrites nifty_core version
    """
        Replaces NIFTY's exp-function.

        See Also
        --------
        nifty.nifty_core.exp

        Notes
        -----
        This exp-function restricts input values to be smaller than 709 in
        order to avoid infinite return values.

    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.exp(np.minimum(709,x.val)),target=x.target) ## prevent overflow
    else:
        return np.exp(np.minimum(709,np.array(x))) ## prevent overflow

def log(x,base=None): ## overwrites nifty_core version
    """
        Replaces NIFTY's log-function.

        See Also
        --------
        nifty.nifty_core.log

        Notes
        -----
        This log-function restricts input values to be larger than 1E-323 in
        order to avoid infinite return values.

    """
    ## TODO: positivity check (seems unnecessary)

    if(base is None):
        if(isinstance(x,field)):
            return field(x.domain,val=np.log(np.maximum(1E-323,x.val)),target=x.target) ## prevent underflow
        else:
            return np.log(np.array(np.maximum(1E-323,x))) ## prevent underflow

    base = np.array(base)
    if(np.all(base>0)):
        if(isinstance(x,field)):
            return field(x.domain,val=np.log(np.maximum(1E-323,x.val))/np.log(base).astype(x.domain.datatype),target=x.target) ## prevent underflow
        else:
            return np.log(np.array(np.maximum(1E-323,x)))/np.log(base) ## prevent underflow
    else:
        raise ValueError(about._errors.cstring("ERROR: invalid input basis."))

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class configuration(object):
    """
        D3PO configuration class.

        Its purpose is to read and process a D3PO config file.

        Parameters
        ----------
        filename : string
            Name of the D3PO config file (default: "config").
        workingdirectory : string
            Name of the working directory (default: os.getcwd()).

        See Also
        --------
        problem


        Attributes
        ----------
        wd : string
            Name of the working directory.
        name : string
            Name of the D3PO config file.
        alpha : {scalar, array}
            Model parameter :math:`\\alpha`.
        q : {scalar, array}
            Model parameter :math:`q`.
        sigma : {scalar, array}
            Model parameter :math:`\sigma`.
        b : {scalar, array}
            Effective model parameter :math:`\\beta-1`.
        eta : {scalar, array}
            Model parameter :math:`\eta`.
        ncpu : integer
            Number of CPUs to use for probing.
        nrun : integer
            Number of probes.
        map_mode : integer
            Mode for map reconstruction. Supported modes are

            * map_mode = ``+3`` (s ~ Gibbs, u ~ Gibbs) -- **INACTIVE**
            * map_mode = ``+2`` (s ~ Gibbs, u ~ MAP)
            * map_mode = ``+1`` (s ~ MAP, u ~ MAP)
            * map_mode = ``-1`` (s ~ MAP, no u)
            * map_mode = ``-2`` (s ~ Gibbs, no u)

        tau_mode : integer
            Mode for power spectrum reconstruction. Supported modes are

            * tau_mode = ``+2`` (t ~ critical)
            * tau_mode = ``+1`` (t ~ MAP)
            * tau_mode = ``+0`` (no t)
            * tau_mode = ``-1`` (t ~ MAP, without smoothness)
            * tau_mode = ``-2`` (t ~ Gibbs, without smoothness)

        tau_tol : float
            Tolerance for changes in the logarithmic power spectrum.
        aftermath : {bool, integer}
            Mode for aftermath.
        notes : bool
            Flag for notifications.
        saves : bool
            Flag for intermediate saves.
        seed : object
            Random seed initializer.
        precondition_D : bool
            Flag for preconditioner use.
        kbands : {float, array}
            Array of spectral bands for intial cycles or the inverse number of
            bands.
        gaussmooth : {bool, array}
            False or Gaussian smoothing kernel for diffuse signal gradient.
        s0 : object
            Starting value for ``s_``.
        p0 : object
            Starting value for power spectrum.
        t0 : object
            Starting value for logarithmic power spectrum.
        u0 : object
            Starting value for ``u_``.
        D0 : object
            Starting value for ``D_``.
        F0 : object
            Starting value for ``F_``.
        pindexing : dict
            Dictionary for power indexing.
        pinfering : dict
            Dictionary for power spectrum inference.
        iniSD_s : dict
            Dictionary for initializing SD solver for ``s_``.
        runSD_s: tupel of dicts
            Dictionary for running SD solver for ``s_``.
        iniSD_u : dict
            Dictionary for initializing SD solver for ``u_``.
        runSD_u: tupel of dicts
            Dictionary for running SD solver for ``u_``.
        useCGprobing_Dkk : tupel of dicts
            Dictionary for running CG solver for ``D`` in the harmonic basis.
        useCGprobing_Dxx : tupel of dicts
            Dictionary for running CG solver for ``D`` in the image basis.
        useCGprobing_Fxx : tupel of dicts
            Dictionary for running CG solver for ``F`` in the image basis.
        useCG_F : dict
            Dictionary for running CG solver for ``F`` on individual image pixels.

    """
    def __init__(self,filename="config",workingdirectory=os.getcwd()):
        """
            Initializes the configuration.

            Its purpose is to read and process a D3PO config file.

            Parameters
            ----------
            filename : string
                Name of the D3PO config file (default: "config").
            workingdirectory : string
                Name of the working directory (default: os.getcwd()).

            Raises
            ------
            IOError
                If either the file or the directory does not exist.
            ValueError
                If a model parameter is out of bounds.

        """
        ## check working directory
        wd = os.path.realpath(workingdirectory)
        if(not os.path.exists(wd)):
            parent = os.path.abspath(os.path.join(wd,os.pardir))
            if(os.path.exists(parent)):
                os.mkdir(wd)
            else:
                raise IOError(about._errors.cstring("ERROR: '"+wd+"' nonexisting."))
        self.wd = wd
        ## check configuration file
        if(not os.path.isfile(filename)):
            purefile = os.path.basename(filename)
            if(os.path.isfile(os.path.join(self.wd,purefile))):
                filename = os.path.join(self.wd,purefile)
            elif(os.path.isfile(os.path.join(os.getcwd(),purefile))):
                filename = os.path.join(os.getcwd(),purefile)
            else:
                raise IOError(about._errors.cstring("ERROR: '"+filename+"' nonexisting."))
        self.name = filename
        ## read configuration file as module
        with open(filename,'r') as configfile:
            cfg = imp.new_module("cfg")
            #exec(configfile.read(),cfg.__dict__) ## Python 3.x
            exec configfile.read() in cfg.__dict__ ## Python 2.x

        ## check model parameters
        if(np.isscalar(cfg.alpha)):
            if(cfg.alpha<1):
                raise ValueError(about._errors.cstring("ERROR: invalid alpha ( "+str(cfg.alpha)+" < 1 )."))
            self.alpha = cfg.alpha
        else:
            alpha = np.asarray(cfg.alpha,order='C')
            if(np.any(alpha<=0)):
                raise ValueError(about._errors.cstring("ERROR: invalid alpha ( "+str(alpha.min())+" < 1 )."))
            self.alpha = alpha.flatten(order='C')
        if(np.isscalar(cfg.q)):
            if(cfg.q<0):
                raise ValueError(about._errors.cstring("ERROR: negative q."))
            self.q = cfg.q
        else:
            q = np.asarray(cfg.q,order='C')
            if(np.any(q<=0)):
                raise ValueError(about._errors.cstring("ERROR: negative q."))
            self.q = q.flatten(order='C')
        if(np.isscalar(cfg.sigma)):
            if(cfg.sigma<=0):
                raise ValueError(about._errors.cstring("ERROR: nonpositive sigma."))
            self.sigma = cfg.sigma
        else:
            sigma = np.asarray(cfg.sigma,order='C')
            if(np.any(sigma<=0)):
                raise ValueError(about._errors.cstring("ERROR: nonpositive sigma."))
            self.sigma = sigma.flatten(order='C')
        if(np.isscalar(cfg.beta)):
            if(cfg.beta<1):
                raise ValueError(about._errors.cstring("ERROR: invalid beta ( "+str(cfg.beta)+" < 1 )."))
            self.b = cfg.beta-1
        else:
            beta = np.asarray(cfg.beta,order='C')
            if(np.any(beta<=1)):
                raise ValueError(about._errors.cstring("ERROR: invalid beta ( "+str(beta.min())+" < 1 )."))
            self.b = beta.flatten(order='C')-1
        if(np.isscalar(cfg.eta)):
            if(cfg.eta<0):
                raise ValueError(about._errors.cstring("ERROR: negative eta."))
            self.eta = cfg.eta
        else:
            eta = np.asarray(cfg.eta,order='C')
            if(np.any(eta<=0)):
                raise ValueError(about._errors.cstring("ERROR: negative eta."))
            self.eta = eta.flatten(order='C')
        ## check multiprocessing parameters
        if(cfg.ncpu<1)or(cfg.nper<1):
            raise ValueError(about._errors.cstring("ERROR: nonpositive multiprocessing parameter."))
        self.ncpu = int(cfg.ncpu)
        self.nper = int(cfg.nper)

        ## compute modes
        ###self.map_mode = (1-2*int(bool(cfg.NO_u)))*(1+int(not cfg.MAP_s)+int(not bool(cfg.NO_u)*bool(cfg.MAP_s)*bool(cfg.MAP_u)))
        self.map_mode = (1-2*int(bool(cfg.NO_u)))*(1+int(not cfg.MAP_s))
        ## map modes: 3 ~ (Gibbs_s,Gibbs_u)
        ##            2 ~ (Gibbs_s,MAP_u)
        ##            1 ~ (MAP_s,MAP_u)
        ##           -1 ~ only MAP_s
        ##           -2 ~ only Gibbs_s
        self.tau_mode = int(not cfg.NO_t)*(1-2*int(not cfg.apply_smoothness))*(1+int(not cfg.MAP_t))
        ## tau modes: 2 ~ Gibbs_t
        ##            1 ~ MAP_t
        ##            0 ~ NO_t
        ##           -1 ~ MAP_t   (without smoothness)
        ##           -2 ~ Gibbs_t (without smoothness)

        ## set remaining flags and values
        self.tau_tol = cfg.tau_tol
        self.aftermath = int(cfg.aftermath).__neg__() ## negative cycles
        self.notes = cfg.notes
        self.saves = cfg.saves
        self.seed = cfg.seed
        self.precondition_D = cfg.precondition_D
        self.kbands = cfg.nb
        self.gaussmooth = cfg.sb

        ## set starting values
        self.s0 = cfg.s0
        self.p0 = cfg.p0
        self.t0 = cfg.t0
        self.u0 = cfg.u0
        self.D0 = cfg.D0
        self.F0 = cfg.F0

        ## build dictionaries
        self.pindexing = {"log":cfg.log,"nbin":cfg.nbin,"binbounds":cfg.binbounds}
        self.pinfering = {"q":self.q,"alpha":self.alpha,"perception":cfg.perception,"smoothness":cfg.apply_smoothness,"var":self.sigma**2,"force":cfg.force_smoothness}
        self.iniSD_s = cfg.iniSD_s
        self.runSD_s = ({"tol":min(1E-1,1E-1*cfg.map_tol)},{"tol":cfg.map_tol})
        self.runSD_s[0].update(cfg.runSD_s[0])
        self.runSD_s[1].update(cfg.runSD_s[1])
        self.iniSD_u = cfg.iniSD_u
        self.runSD_u = ({"tol":min(1E-1,1E-1*cfg.map_tol)},{"tol":cfg.map_tol})
        self.runSD_u[0].update(cfg.runSD_u[0])
        self.runSD_u[1].update(cfg.runSD_u[1])
        cfg.iniCGprobing.update(cfg.runCGprobing)
        cfg.iniCGprobing_D.update(cfg.iniCGprobing,ncpu=self.ncpu,nper=self.nper)
        cfg.runCGprobing_Dkk[0].update(cfg.iniCGprobing_D)
        cfg.runCGprobing_Dkk[1].update(cfg.iniCGprobing_D)
        cfg.runCGprobing_Dxx[0].update(cfg.iniCGprobing_D)
        cfg.runCGprobing_Dxx[1].update(cfg.iniCGprobing_D)
        cfg.iniCGprobing_F.update(cfg.iniCGprobing,ncpu=self.ncpu,nper=self.nper)
        cfg.runCGprobing_Fxx[0].update(cfg.iniCGprobing_F)
        cfg.runCGprobing_Fxx[1].update(cfg.iniCGprobing_F)
        self.useCGprobing_Dkk = (cfg.runCGprobing_Dkk[0],cfg.runCGprobing_Dkk[1])
        self.useCGprobing_Dxx = (cfg.runCGprobing_Dxx[0],cfg.runCGprobing_Dxx[1])
        self.useCGprobing_Fxx = (cfg.runCGprobing_Fxx[0],cfg.runCGprobing_Fxx[1])
        self.useCG_F = {kk:self.useCGprobing_Fxx[1][kk] for kk in ["W","spam","reset","note","x0","tol","clevel","limii"] if self.useCGprobing_Fxx[1].has_key(kk)}

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _update(self,problem):
        """
            Updates the configuration.

            See Also
            --------
            problem

            Notes
            -----
            This workaround updates the configuration; e.g., with problem
            specific parameters such as ``kindex``.

        """
        ## check and update model parameter shapes
        if(not np.isscalar(self.alpha)):
            if(len(self.alpha)!=len(problem.k.power_indices["kindex"])):
                raise ValueError(about._errors.cstring("ERROR: misshaped alpha."))
        if(not np.isscalar(self.q)):
            if(len(self.q)!=len(problem.k.power_indices["kindex"])):
                raise ValueError(about._errors.cstring("ERROR: misshaped q."))
        if(not np.isscalar(self.sigma)):
            if(len(self.sigma)!=len(problem.k.power_indices["kindex"])):
                raise ValueError(about._errors.cstring("ERROR: misshaped sigma."))
        if(not np.isscalar(self.b)):
            if(self.b.size!=problem.z.dim(split=False)):
                raise ValueError(about._errors.cstring("ERROR: misshaped beta."))
            self.b.shape = tuple(problem.z.dim(split=True))
        if(not np.isscalar(self.eta)):
            if(self.eta.size!=problem.z.dim(split=False)):
                raise ValueError(about._errors.cstring("ERROR: misshaped eta."))
            self.eta.shape = tuple(problem.z.dim(split=True))
        ## update dictionaries
        newCGprobing = {}
        newCGprobing.update(domain=problem.z,target=problem.k)
        self.useCGprobing_Fxx[0].update(newCGprobing)
        self.useCGprobing_Fxx[1].update(newCGprobing)
        if(self.precondition_D):
            newCGprobing.update(W=problem.S)
        self.useCGprobing_Dxx[0].update(newCGprobing)
        self.useCGprobing_Dxx[1].update(newCGprobing)
        newCGprobing.update(domain=problem.k,target=problem.z)
        self.useCGprobing_Dkk[0].update(newCGprobing)
        self.useCGprobing_Dkk[1].update(newCGprobing)
        ## evaluate starting spectra
        if(callable(self.p0)): ## un-deprecated in Python 3.2+
            k_ = np.r_[1,problem.k.power_indices["kindex"][1:]] ## k[0] == 0
            self.p0 = self.p0(k_)
        if(callable(self.t0)): ## un-deprecated in Python 3.2+
            k_ = np.r_[1,problem.k.power_indices["kindex"][1:]] ## k[0] == 0
            self.t0 = self.t0(k_)
            if(self.p0 is None):
                self.p0 = np.log(self.t0)
        if(self.p0 is None):
            self.p0 = np.r_[1,(problem.k.power_indices["kindex"][1]/problem.k.power_indices["kindex"][1:])**problem.k.dim(split=True).size] ## kindex^(-dim)
        ## update remaining values
        if(np.isscalar(self.kbands)):
            if(self.kbands==0):
                self.kbands = np.arange(2,len(problem.k.power_indices["kindex"]),1,dtype=np.int)
            else:
                nk = np.round(1/float(self.kbands),decimals=3)
                dk = problem.k.power_indices["kindex"][-1]/problem.k.power_indices["kindex"][2]
                self.kbands = np.unique(np.searchsorted(problem.k.power_indices["kindex"],problem.k.power_indices["kindex"][2]*dk**np.arange(0,1+0.5*nk,nk),side="left"),return_index=False,return_inverse=False)
        else:
            self.kbands = np.minimum(np.maximum(1,np.array(self.kbands).astype(np.int)),len(problem.k.power_indices["kindex"])-1)
        if(self.gaussmooth is not False):
            self.gaussmooth = 10**(-self.gaussmooth*(problem.k.power_indices["kindex"]/problem.k.power_indices["kindex"][-1])**2)[problem.k.power_indices["pindex"]]
        ## add more (e.g., limii as a function of dimensionality)

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class Ms_operator(operator):
    """
        Operator comprising the likelihood contribution to the diffuse signal's
        covariance operator.

        See Also
        --------
        nifty.operator

        Notes
        -----
        Confer D3PO reference equation (B.2) or (B.5).

    """
    def _multiply(self,x,**kwargs):
        e = exp(self.para.s_eff())
        if(self.para.config.map_mode>0):
            l = self.para.R(e+exp(self.para.u_eff()))
        else:
            l = self.para.R(e)
        ll = problem._fix_inv0(l)
        y = self.para.R.adjoint_times(1-self.para.d*ll,target=self.para.k)
        y *= x
        dd = ll**2
        dd *= self.para.R(e*x)
        dd *= self.para.d
        y += self.para.R.adjoint_times(dd,target=self.para.z) ## target irrelevant
        y *= e
        return y

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

class Ds_operator(propagator_operator):
    """
        The diffuse signal's covariance operator D^(s), here ``D``.

        See Also
        --------
        nifty.nifty_tools.propagator_operator

        Notes
        -----
        This operator is derived from the :py:class:`propagator_operator`.
        It uses a conjugate gradient for (normal) application, and involves
        Levenberg damping to ensure positive definiteness.
        Confer D3PO reference equation (B.2) or (B.5).

    """
    _L = mv('d',0) ## Levenberg damping

    def reset_L(self):
        """
            Resets the Levenberg damping term.

        """
        self._L.acquire()
        self._L.value = 0
        self._L.release()

    def _set_L(self,newvalue):
        self._L.acquire()
        self._L.value = newvalue
        self._L.release()

    def _inverse_multiply_1(self,x,**kwargs): ## > applies A1 + A2 in self.codomain
        y = self._A2(x.transform(self.domain)).transform(self.codomain)
        y += self._A1(x)
        ## enforce positivity
        dot = self.codomain.calc_dot(x.val,y.val).real
        if(np.signbit(dot)):
            self._set_L(max(self._L.value,-1.01*dot/self.codomain.calc_dot(x.val,x.val).real)) ## non-bare(!)
        y += x*self._L.value
        return y

    def _inverse_multiply_2(self,x,**kwargs): ## > applies A1 + A2 in self.domain
        y = self._A1(x.transform(self.codomain)).transform(self.domain)
        y += self._A2(x)
        ## enforce positivity
        dot = self.domain.calc_dot(x.val,y.val).real
        if(np.signbit(dot)):
            self._set_L(max(self._L.value,-1.01*dot/self.domain.calc_dot(x.val,x.val).real)) ## non-bare(!)
        y += x*self._L.value
        return y

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

class Du_operator(invertible_operator):
    """
        The point-like signal's covariance operator D^(u), here ``F``.

        See Also
        --------
        nifty.nifty_tools.invertible_operator

        Notes
        -----
        This operator is derived from the :py:class:`invertible_operator`.
        It uses a conjugate gradient for (normal) application, and involves
        Levenberg damping to ensure positive definiteness.
        Confer D3PO reference equation (B.3) or (B.6).

    """
    _L = mv('d',0) ## Levenberg damping

    def reset_L(self):
        """
            Resets the Levenberg damping term.

        """
        self._L.acquire()
        self._L.value = 0
        self._L.release()

    def _set_L(self,newvalue):
        self._L.acquire()
        self._L.value = newvalue
        self._L.release()

    def _inverse_multiply(self,x,**kwargs):
        if(self.para.config.map_mode==3):
            e = exp(self.para.u_)
            f = exp(0.5*self.para.F_)
            ef = e*f
            l = self.para.R(exp(self.para.s_eff())+ef)
            ll = problem._fix_inv0(l)
            y = self.para.R.adjoint_times(1-self.para.d*ll,target=self.para.k)
            y *= x
            dd = ll**2
            dd *= self.para.R(ef*x)
            dd *= self.para.d
            y += self.para.R.adjoint_times(dd,target=self.para.z) ## target irrelevant
            y *= ef
            pp = self.para.config.eta/e
            pp *= f
            pp *= x
            y += pp
            return y
        e = exp(self.para.u_)
        l = self.para.R(exp(self.para.s_eff())+e)
        ll = problem._fix_inv0(l)
        y = self.para.R.adjoint_times(1-self.para.d*ll,target=self.para.k)
        y *= x
        dd = ll**2
        dd *= self.para.R(e*x)
        dd *= self.para.d
        y += self.para.R.adjoint_times(dd,target=self.para.z) ## target irrelevant
        y *= e
        pp = self.para.config.eta/e
        pp *= x
        y += pp
        return y

    def inverse_times(self,x,**kwargs):
        """
            Applies the inverse of ``F`` to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Input object.

            Returns
            -------
            y : field
                Mapped field.

            Other Parameters
            ----------------
            force : bool
                Indicates wheter to return a field instead of ``None`` in case
                the conjugate gradient fails.
            W : {operator, function}, *optional*
                Operator `W` that is a preconditioner on `A` and is applicable to a
                field (default: None).
            spam : function, *optional*
                Callback function which is given the current `x` and iteration
                counter each iteration (default: None).
            reset : integer, *optional*
                Number of iterations after which to restart; i.e., forget previous
                conjugated directions (default: sqrt(b.dim())).
            note : bool, *optional*
                Indicates whether notes are printed or not (default: False).
            x0 : field, *optional*
                Starting guess for the minimization.
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by current relative
                residual (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 1).
            limii : integer, *optional*
                Maximum number of iterations performed (default: 10 * b.dim()).

        """
        ## check whether self-inverse
        if(self.sym)and(self.uni):
            return self.times(x,**kwargs)

        ## prepare
        x_ = self._briefing(x,self.target,True)
        ## apply operator
        y_ = self._inverse_multiply(x_,**kwargs)
        ## evaluate
        y_ = self._debriefing(x,y_,self.domain,True)

        ## enforce positivity
        dot = self.domain.calc_dot(x_.val,y_.val).real
        if(np.signbit(dot)):
            self._set_L(max(self._L.value,-1.01*dot/self.domain.calc_dot(x_.val,x_.val).real)) ## non-bare(!)
        y_ += x_*self._L.value
        return y_

##-----------------------------------------------------------------------------



##=============================================================================

class problem(object):
    """
        D3PO problem class.

        Its purpose is to condense a D3PO problem and provide a solver.

        Parameters
        ----------
        R : operator
            Response operator.
        configfile : string
            Name of the D3PO config file (default: "config").
        workingdirectory : string
            Name of the working directory (default: os.getcwd()).

        See Also
        --------
        demo.py

        Examples
        --------
        >>> import d3po
        >>> p = d3po.problem([RESPONSE_OPERATOR],[CONFIG_FILE])
        >>> p.solve([DATA_ARRAY])

        Attributes
        ----------
        R : operator
            Response operator.
        d : field
            Data vector
        config : configuration
            Configuration instance.
        z : space
            Image space.
        k : space
            Harmonic space.
        `s_` : field
             Current diffuse signal field.
        `u_` : field
             Current point-like signal field.
        `D_` : field
             Current diagonal of the diffuse covariance operator.
        `F_` : field
             Current diagonal of the point-like covariance operator.
        S : operator
            Prior covariance of the (current) diffuse signal field.
        Sk : operator
            Projection operator.
        M : operator
            Ms_operator instance.
        D : operator
            Ds_operator instance.
        F : operator
            Du_operator instance.
        note : notification
            Notification instance.
        _restrict : {bool, array}
            Dummy for restrictions.
        _io : array
            Dummy for IO indices

    """
    def __init__(self,R,configfile="config",workingdirectory=os.getcwd()):
        """
            Initializes the problem.

            Parameters
            ----------
            R : operator
                Response operator.
            configfile : string
                Name of the D3PO config file (default: "config").
            workingdirectory : string
                Name of the working directory (default: os.getcwd()).

            Raises
            ------
            TypeError
                If the response (or attributes thereof) are invalid.

        """
        ## check response
        if(not isinstance(R,operator)):
            raise TypeError(about._errors.cstring("ERROR: invalid response."))
        if(not hasattr(R,"identify_io_pixels")):
            about.warnings.cprint("WARNING: response misses attribute 'identify_io_pixels'.")
        if(not hasattr(R,"mask_out_pixels")):
            about.warnings.cprint("WARNING: response misses attribute 'mask_out_pixels'.")
        else:
            if(not isinstance(R.mask_out_pixels,(bool,switch))):
                raise TypeError(about._errors.cstring("ERROR: invalid response attribute 'mask_out_pixels'."))
        self.R = R
        self.d = None
        ## check config
        self.config = configuration(configfile,workingdirectory)
        ## set ...
        ## ... signal spaces
        self.z = self.R.domain
        self.k = self.z.get_codomain()
        self.k.set_power_indices(**self.config.pindexing)
        ## ... (temporary) fields
        self.s_ = None
        self.u_ = None
        self.D_ = None
        self.F_ = None
        ## ... operators
        self.S = power_operator(self.k)
        self.Sk = self.S.get_projection_operator()
        self.M = Ms_operator(self.z,imp=True,para=self)
        self.D = Ds_operator(S=self.S,M=self.M)
        self.F = Du_operator(self.z,imp=True,para=self)
        ## ... notifications
        self.note = notification(default=self.config.notes)
        ## ... hidden variables
        self._restrict = False ## required in Eg?-functions
        self._io = self._io = np.array([[],[]],dtype=np.int) ## 2D arry of raveled IO indices
        ## update config
        self.config._update(self)
        ## copy config
        if(self.config.saves is not False):
            if(not os.path.exists(os.path.join(self.config.wd,"d3po_tmp"))):
                os.mkdir(os.path.join(self.config.wd,"d3po_tmp"))
        if(os.path.exists(os.path.join(self.config.wd,os.path.basename(self.config.name)))):
            about.warnings.cprint("WARNING: overwriting '"+os.path.join(self.config.wd,os.path.basename(self.config.name))+"'.")
        shutil.copy(self.config.name,self.config.wd)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def _fix_log0(x,fix=1,base=None):
        return field(x.domain,val=log(np.ma.filled(np.ma.masked_where(x.val==0,x.val,copy=False),fill_value=fix),base=base),target=x.target)
        #return field(x.domain,val=log(np.where(x.val==0,fix,x.val),base=base),target=x.target)

    @staticmethod
    def _fix_inv0(x,fix=0,dividend=1):
        return field(x.domain,val=np.ma.filled(dividend/np.ma.masked_where(x.val==0,x.val,copy=False),fill_value=fix),target=x.target)
        #return field(x.domain,val=np.where(x.val==0,fix,dividend/x.val),target=x.target)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def s_eff(self,regularize=None):
        """
            Computes the effective diffuse contribution.

        """
        if(abs(self.config.map_mode)>1)or(regularize==True):
            return self.s_+0.5*self.D_
        else:
            return self.s_

    def u_eff(self,regularize=None):
        """
            Computes the effective point-like contribution.

        """
        if(self.config.map_mode==3)or(regularize==True):
            return self.u_+0.5*self.F_
        else:
            return self.u_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calc_likelihood(self,l):
        """
            Computes the (unnormalized) Poissonian likelihood.

            Parameters
            ----------
            l : field
                Expected or most likely number of photon counts.

            Returns
            -------
            E : float
                Likelihood.

            Notes
            -----
            Confer D3PO reference equation (9).

        """
        E = l.dot(1)
        E -= self._fix_log0(l).dot(self.d)
        return E

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calc_Eg_H(self,s):
        """
            Computes the Hamiltonian and its gradient with respect to ``s``.

            Notes
            -----
            Confer D3PO reference equation (31) and (32) disregarding ``u``.

        """
        e = exp(s)
        l = self.R(e)
        ll = self._fix_inv0(l,dividend=self.d.val)
        g = self.R.adjoint_times(1-ll,target=self.k)
        g *= e
        a = s.transform()
        Sa = self.S.inverse_times(a)
        g += Sa.transform()
        E = self._calc_likelihood(l)
        E += 0.5*a.dot(Sa).real
        return E,g

    def _calc_Eg_G(self,s,D):
        """
            Computes the Gibbs free energy and its gradient with respect to ``s``.

            Notes
            -----
            Confer D3PO reference equation (48) and (49) disregarding ``u``.

        """
        e = exp(s+0.5*D)
        l = self.R(e)
        ll = self._fix_inv0(l,dividend=self.d.val)
        g = self.R.adjoint_times(1-ll,target=self.k)
        g *= e
        a = s.transform()
        Sa = self.S.inverse_times(a)
        g += Sa.transform()
        E = self._calc_likelihood(l)
        E += 0.5*a.dot(Sa).real
        return E,g

    def _calc_Eg_Hs(self,s,u_eff):
        """
            Computes the Hamiltonian and its gradient with respect to ``s``.

            Notes
            -----
            Confer D3PO reference equation (31) and (32).

        """
        e = exp(s)
        l = self.R(e+exp(u_eff))
        ll = self._fix_inv0(l,dividend=self.d.val)
        g = self.R.adjoint_times(1-ll,target=self.k)
        g *= e
        a = s.transform()
        Sa = self.S.inverse_times(a)
        g += Sa.transform()
        E = self._calc_likelihood(l)
        E += 0.5*a.dot(Sa).real
        return E,g

    def _calc_Eg_Gs(self,s,u_eff,D):
        """
            Computes the Gibbs free energy and its gradient with respect to ``s``.

            Notes
            -----
            Confer D3PO reference equation (48) and (49).

        """
        e = exp(s+0.5*D)
        l = self.R(e+exp(u_eff))
        ll = self._fix_inv0(l,dividend=self.d.val)
        g = self.R.adjoint_times(1-ll,target=self.k)
        g *= e
        a = s.transform()
        Sa = self.S.inverse_times(a)
        g += Sa.transform()
        E = self._calc_likelihood(l)
        E += 0.5*a.dot(Sa).real
        return E,g

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def Egs(self,s):
        """
            Evaluates energy and gradient for the diffuse signal field.

        """
        ## compute energy and gradient
        if(self.config.map_mode==1):
            E,g = self._calc_Eg_Hs(s,self.u_)
        elif(self.config.map_mode==2):
            E,g = self._calc_Eg_Gs(s,self.u_,self.D_)
        elif(self.config.map_mode==3):
            E,g = self._calc_Eg_Gs(s,self.u_+0.5*self.F_,self.D_)
        elif(self.config.map_mode==-1):
            E,g = self._calc_Eg_H(s)
        elif(self.config.map_mode==-2):
            E,g = self._calc_Eg_G(s,self.D_)
        else:
            raise ValueError(about._errors.cstring("ERROR: invalid map_mode '"+str(self.config.map_mode)+"'."))
        ## restrict gradient
        if(self._restrict is not False):
            g = (g.transform()*self._restrict).transform()
        elif(self.config.gaussmooth is not False):
            g = (g.transform()*self.config.gaussmooth).transform()
        ## return energy and gradient
        return E,g

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calc_Eg_Hu(self,s_eff,u,energy=True):
        """
            Computes the Hamiltonain and its gradient with respect to ``u``.

            Notes
            -----
            Confer D3PO reference equation (31) and (33).

        """
        e = exp(u)
        l = self.R(exp(s_eff)+e)
        ll = self._fix_inv0(l,dividend=self.d.val)
        g = self.R.adjoint_times(1-ll,target=self.k)
        g *= e
        e **= -1
        g -= e*self.config.eta
        g += self.config.b
        if(energy):
            E = self._calc_likelihood(l)
            E += e.dot(self.config.eta)
            E += u.dot(self.config.b)
            return E,g
        else:
            return g

    def _calc_Eg_Gu(self,s_eff,u,F,energy=True):
        """
            Computes the Gibbs free energy and its gradient with respect to ``u``.

            Notes
            -----
            Confer D3PO reference equation (48) and (50).

        """
        e = exp(u)
        f = exp(0.5*F)
        ef = e*f
        l = self.R(exp(s_eff)+ef)
        ll = self._fix_inv0(l,dividend=self.d.val)
        g = self.R.adjoint_times(1-ll,target=self.k)
        g *= ef
        f /= e
        g -= f*self.config.eta
        g += self.config.b
        if(energy):
            E = self._calc_likelihood(l)
            E += f.dot(self.config.eta)
            E += u.dot(self.config.b)
            return E,g
        else:
            return g

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def Egu(self,u):
        """
            Evaluates energy and gradient for the point-like signal field.

        """
        ## compute energy and gradient
        if(self.config.map_mode==3):
            E,g = self._calc_Eg_Gu(self.s_eff(),u,self.F_)
        else:
            E,g = self._calc_Eg_Hu(self.s_+0.5*self.D_,u)
        ## restrict gradient
        if(self._restrict is not False):
            g *= self._restrict
        ## return energy and gradient
        return E,g

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calc_Dkk(self):
        """
            Calculates the operator ``D`` in the harmonic basis using CG.

        """
        ## reset Levenberg damping
        self.D.reset_L()
        ## initial and final probing
        diag = self.D.diag(bare=False,**self.config.useCGprobing_Dkk[0])
        if(diag is not None):
            diag = self.D.diag(bare=False,**self.config.useCGprobing_Dkk[1])
        ## return diagonal operator
        if(diag is None):
            about.warnings.cprint("WARNING: divergent diagonal reset to 0.")
            return 0
        else:
            return diagonal_operator(self.k,diag=diag,bare=False)

    def _calc_Dxx(self,include_old=False):
        """
            Calculates the diagonal of the operator ``D`` in the image basis using CG.

        """
        ## reset Levenberg damping
        self.D.reset_L()
        ## initial and final probing
        diag = self.D.diag(bare=True,**self.config.useCGprobing_Dxx[0])
        if(diag is not None):
            diag = self.D.diag(bare=True,**self.config.useCGprobing_Dxx[1])
            if(include_old):
                diag += self.D_.val
                diag *= 0.5
        ## return diagonal as field
        if(diag is None):
            about.warnings.cprint("WARNING: divergent diagonal reset to 0.")
            return field(self.z,val=None,target=self.k),False
        if(np.any(diag<0)):
            about.warnings.cprint("WARNING: divergent element(s) of Dxx reset to 0.") ## increase nrun?
            return field(self.z,val=np.maximum(0,diag),target=self.k),False
        else:
            return field(self.z,val=diag,target=self.k),(not self.D._L.value)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calc_Fxx(self,include_old=False):
        """
            Calculates the diagonal of the operator ``F`` in image basis using CG.

        """
        ## reset Levenberg damping
        self.F.reset_L()
        ## initial and final probing
        diag = self.F.diag(bare=True,**self.config.useCGprobing_Fxx[0])
        if(diag is not None):
            diag = self.F.diag(bare=True,**self.config.useCGprobing_Fxx[1])
            if(include_old):
                diag += self.F_.val
                diag *= 0.5
        ## return diagonal as field
        if(diag is None):
            about.warnings.cprint("WARNING: divergent diagonal reset to 0.")
            return field(self.z,val=None,target=self.k),False
        elif(np.any(diag<0)):
            about.warnings.cprint("WARNING: divergent element(s) of Fxx reset to 0.") ## increase nrun?
            return field(self.z,val=np.maximum(0,diag),target=self.k),False
        else:
            return field(self.z,val=diag,target=self.k),bool(self.F._L.value)

    def _calc_Fii(self):
        """
            Calculates the diagonal of the operator ``F`` in image basis using CG,
            evaluated at certain positions only.

        """
        ## initial diagonal
        diag = np.zeros(self.z.dim(split=True))
        ## volume elements
        unit = 1/self.z.get_meta_volume(total=False).flatten(order='C')
        ## iterate over point sources
        L = False
        for ii in self._io[0]:
            ## reset Levenberg damping
            self.F.reset_L()
            ## define unit vector
            e = field(self.z,target=self.k)
            ind = np.unravel_index(ii,self.z.dim(split=True))
            e.val[ind] = unit[ii] ## 1/V
            ## evaluate Fii
            Fe = self.F(e,**self.config.useCG_F)
            if(Fe is None)or(self.F._L.value>0):
                pass
            else:
                diag[ind] = e.dot(Fe) ## bare(!)
                L += bool(self.F._L.value)
        ## return diagonal as field
        if(np.any(diag<0)):
            about.warnings.cprint("WARNING: divergent element(s) of Fxx reset to 0.")
            return field(self.z,val=np.maximum(0,diag),target=self.k),False
        else:
            return field(self.z,val=diag,target=self.k),(not L)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _infer_p(self,cycle):
        """
            Infers the current power spectrum using *critical smooth filtering*.

        """
        ## precompute Dkk
        if(abs(self.config.tau_mode)==1)or(not cycle): ## no Dkk correction in cycle 0
            Dkk = None
        elif(abs(self.config.tau_mode)==2):
            Dkk = self._calc_Dkk()
        else:
            raise ValueError(about._errors.cstring("ERROR: invalid map_mode '"+str(self.config.map_mode)+"'."))
        ## return infered power
        return infer_power(self.s_,domain=self.k,Sk=self.Sk,D=Dkk,bare=True,**self.config.pinfering)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _optimize_s(self,cycle,bband=None,save=True,solve_s=None):
        """
            Optimizes the current diffuse signal field using SD.

        """
        convergence_s = True
        if(cycle>=0):
            ## set restriction
            if(cycle<2)and(bband is not None):
                if(self.config.gaussmooth is not False):
                    self._restrict = self.config.gaussmooth**((self.k.power_indices["kindex"][-1]/self.k.power_indices["kindex"][bband])**2)
                else:
                    self._restrict = np.where(self.k.power_indices["pindex"]>bband,0,1)
            ## solve s
            if(solve_s is None):
                solve_s = steepest_descent(self.Egs,**self.config.iniSD_s)
            self.s_,convergence_s = solve_s(self.s_,**self.config.runSD_s[int(cycle>1)])
            self.note.cprint("s-convergence : "+str(bool(convergence_s)))
            if(save):
                self._save(char='s',cycle=cycle)
        return bool(convergence_s)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _optimize_t(self,cycle,bband=None,save=True):
        """
            Optimizes the current power spectrum.

        """
        convergence_t = True
        if(cycle>=0)and(self.config.tau_mode):
            p_ = self._infer_p(cycle)
            if(cycle<2)and(bband is not None):
                p_ = np.r_[p_[:bband],p_[bband-1]*self.config.p0[bband:]] ## alter front part, fix rare
                ## ignore convergence
            else:
                ## check convergence
                convergence_t = (max(np.absolute(log(p_/self.S.get_power(),base=self.S.get_power())))<self.config.tau_tol)
                self.note.cprint("t-convergence : "+str(bool(convergence_t)))
            self.S.set_power(p_,bare=True)
            if(save):
                self._save(char='p',cycle=cycle)
        return convergence_t

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _optimize_u(self,cycle,raising=False,solve_u=None):
        """
            Optimizes the current point-like signal field using SD.

        """
        convergence_u = True
        if(cycle>=0)and(self.config.map_mode>0):
            if(cycle<2)or(raising): ## turns (eventually) off
                ## raise u
                raising = self._raise_u(cycle)
            ## set restriction
            if(cycle<2):
                if(self._io.size>0):
                    self._restrict = np.zeros(self.z.dim(split=False),dtype=np.int)
                    self._restrict[self._io] = np.ones(len(self._io[0]),dtype=np.int)
                else:
                    self._restrict = False
            ## solve u
            if(solve_u is None):
                solve_u = steepest_descent(self.Egu,**self.config.iniSD_u)
            self.u_,convergence_u = solve_u(self.u_,**self.config.runSD_u[int(cycle>1)])
            self.note.cprint("u-convergence : "+str(bool(convergence_u)))
            self._save(char='u',cycle=cycle)
        return bool(convergence_u),raising

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _optimize_D(self,cycle):
        """
            Optimizes the current diffuse signal (co)variance.

        """
        convergence_D = True
        if(cycle>1)and(abs(self.config.map_mode)>1):
            ## compute Dxx
            self.D_,convergence_D = self._calc_Dxx()
            self._save(char='D',cycle=cycle)
        elif(cycle<0)and(self.config.aftermath):
            ## compute Dxx
            self.D_,convergence_D = self._calc_Dxx(include_old=(self.config.aftermath>0))
            self._save(char='D',cycle=cycle)
        return bool(convergence_D)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _optimize_F(self,cycle):
        """
            Optimizes the current point-like signal (co)variance.

        """
        convergence_F = True
        ## check cycle and mode
        if(cycle>1)and(self.config.map_mode==3):
            ## compute Fxx
            self.F_,convergence_F = self._calc_Fxx()
            self._save(char='F',cycle=cycle)
        elif(cycle<0)and(self.config.aftermath):
            if(self.config.map_mode==3):
                ## compute Fxx
                self.F_,convergence_F = self._calc_Fxx(include_old=(self.config.aftermath>0))
                self._save(char='F',cycle=cycle)
            elif(self.config.map_mode>0):
                ## compute Fii
                self.F_,convergence_F = self._calc_Fii()
                self._save(char='F',cycle=cycle)
        return bool(convergence_F)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _infer_s(self,cycle,solve_s=None):
        """
            Infers the current diffuse contributions.

        """
        convergence = True
        if(cycle>1):
            ## optimize (all bands)
            convergence *= self._optimize_s(cycle,solve_s=solve_s)
            self._optimize_D(cycle)
            convergence *= self._optimize_t(cycle)
        elif(cycle>=0):
            ## optimize (iteratively including higher bands)
            for bb in self.config.kbands:
                _save = True#(bb==self.config.kbands[-1])
                convergence *= self._optimize_s(cycle,bband=bb,save=_save,solve_s=solve_s)
                convergence *= self._optimize_t(cycle,bband=bb,save=_save)
        else:
            ## optimize (covariance only)
            convergence *= self._optimize_D(cycle)
        return convergence

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _infer_u(self,cycle,solve_u=None,raising=False):
        """
            Infers the current point-like contributions.

        """
        convergence = True
        if(cycle>=0):
            ## optimize
            convergence,raising = self._optimize_u(cycle,raising=False,solve_u=solve_u)
            self._optimize_F(cycle)
            return convergence,raising
        else:
            ## optimize (covariance only)
            return self._optimize_F(cycle),False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _raise_s(self,cycle,solve_s=None):
        """
            Raises an initial guess for the diffuse signal field.

        """
        ## check whether "raising" is required
        if(self.config.map_mode>0)and(np.all(self.s_.val==0)):
            if(hasattr(self.R,"mask_out_pixels"))and(hasattr(self.R,"mask")):
                ## activate masking
                if(isinstance(self.R.mask_out_pixels,switch)):
                    self.R.mask_out_pixels.on()
                else:
                    self.R.mask_out_pixels = True
                ## optimize (iteratively including higher bands)
                for bb in self.config.kbands:
                    _save = True#(bb==self.config.kbands[-1])
                    self._optimize_s(cycle,bband=bb,save=_save,solve_s=solve_s)
                    self._optimize_t(cycle,bband=bb,save=_save)
                self.s_ += 0.1*self.s_.val.mean() ## overshoot(!)
                ## reset
                self.S.set_power(self.config.p0,bare=True) ## p0 is bare(!)
                ## deactivate masking
                if(isinstance(self.R.mask_out_pixels,switch)):
                    self.R.mask_out_pixels.off()
                else:
                    self.R.mask_out_pixels = False
            else:
                ## guess "dirty" diffuse signal
                R1 = self.R(1)
                rho = self.R.adjoint_times(self.d,target=self.k) ## ~ "dirty" (total) rho
                if(not hasattr(self.R,"den")):
                    about.warnings.cprint("WARNING: response misses attribute 'den'.")
                    rho.weight(power=-1,overwrite=True)
                elif(not self.R.den):
                    rho.weight(power=-1,overwrite=True)
                RR1 = self.R.adjoint_times(R1,target=self.k)
                rho.val *= self._fix_inv0(RR1,fix=1/RR1.val.mean()) ## divide out exposure
                e = field(self.z,target=self.k)
                ind = np.unravel_index(rho.val.argmax(),self.z.dim(split=True),order='C')
                e.val[ind] = 1
                RRe = self.R.adjoint_times(self.R(e),target=self.z)[ind]
                rho *= np.sqrt(RR1[ind]/RRe) ## divide out PSF
                a = self._fix_log0(rho,fix=min(1,np.ma.masked_where(rho.val==0,rho.val,copy=False).min())).transform()
                norm = a.norm(q=2)
                k_cut = sqrt(self.k.power_indices["kindex"][2]*sqrt(self.k.power_indices["kindex"][2]*self.k.power_indices["kindex"][-1])) ## arbitrary
                a *= (self.k.power_indices["kindex"]<=k_cut)[self.k.power_indices["pindex"]] ## cut "high" bands
                a[np.unravel_index(self.k.power_indices["pundex"][0],self.k.dim(split=True))] += 2*(norm-a.norm(q=2)) ## find upper bound by adding power from cut bands (twice)
                self.s_ = a.transform() ## "dirty" s

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _raise_u(self,cycle):
        """
            Raises an initial guess for the point-like signal field.

            Point sources are rasied by increasing ``u`` at the
            most promising positions iteratively.

        """
        indices = ([],[])
        ## reduce data
        s_eff = self.s_eff()
        rs = self.R(exp(s_eff))
        rs += sqrt(rs) ## overshoot(!)
        d = (self.d-rs).val.flatten(order='C')
        ## search
        while(True):
            ## compute "involved" gradient
            if(self.config.map_mode==3):
                involve = (d-self.R(exp(self.u_+0.5*self.F_)).val.flatten(order='C')>0)
                g_ = self._calc_Eg_Gu(s_eff,self.u_,self.F_,energy=False).val.flatten(order='C')
            else:
                involve = (d-self.R(exp(self.u_)).val.flatten(order='C')>0)
                g_ = self._calc_Eg_Hu(s_eff,self.u_,energy=False).val.flatten(order='C')
            if(hasattr(self.R,"identify_io_pixels")):
                if(self.R.identify_io_pixels):
                    g_ *= involve
            ind_u = g_.argmin()
            if(np.signbit(g_[ind_u])): ## most negative gradient ~ most potential point source
                ## unravel index
                ind = np.unravel_index(ind_u,self.z.dim(split=True),order='C')
                ## manage raveled indices
                in_io = (self._io[0]==ind_u)
                if(np.any(in_io)):
                    ind_d = self._io[1][in_io][0]
                else:
                    in_indices = (indices[0]==ind_u)
                    if(np.any(in_indices)):
                        ind_d = indices[1][in_indices.argmax()] ## equivalent to np.array(indices[1])[in_indices][0]
                    else:
                        indices[0].append(ind_u)
                        if(self.config.iniSD_u.get("note",False)):
                            self.note.cflush("\nraised : %3u of %3u"%(len(self._io[0])+len(indices[0]),len(self._io[0])+len(indices[0])+involve.sum()))
                        ## compute data space index
                        ind_d = None
                        if(hasattr(self.R,"identify_io_pixels")):
                            if(self.R.identify_io_pixels):
                                ind_d = ind_u
                        if(ind_d is None):
                            R1 = self.R(1)
                            e = field(self.z,target=self.k)
                            e.val[ind] = 1
                            re = self.R(e)
                            re *= self._fix_inv0(R1)
                            ind_d = re.val.argmax()
                        indices[1].append(ind_d)
                ## raise u
                while(True):
                    self.u_.val[ind] += 1 ## raise ... overshoot(!)
                    if(self.config.map_mode==3):
                        ru = self.R(exp(self.u_+0.5*self.F_))[ind_d]
                    else:
                        ru = self.R(exp(self.u_))[ind_d]
                    if(d[ind_d]<ru):
                        break
            else:
                break
        ## update indices
        if(len(indices[0])>0):
            self._io = np.r_[self._io.T,np.asarray(indices,dtype=np.int,order='C').T].T
            if(self.config.iniSD_u.get("note",False)):
                self.note.cprint("\n... done.")
        ## return raising
        return bool(len(indices[0]))or(cycle<2)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def optimize(self,cycle=0):
        """
            Optimizes all parameters until convergence.

            Parameters
            ----------
            cycle : integer
                Starting cycle (default: 0).

            Notes
            -----
            The global optimization of all problem parameters is done in cycles
            until convergence. Given default configuration: Cycle 0 yields an
            initial `s` estimate, and rough `u`, `s`, and `p` estimate;
            Cycle 1 yields a second rough estimate after forgetting the others;
            Cycle 2+ yield estimates for `u`, `F`, `s`, `p`, and `D`; and the
            aftermath cycles (<0) improve the estimates on `F` and `D`.
            For details ...

            See Also
            --------
            _infer_s, _infer_u

            Raises
            ------
            KeyboardInterrupt
                If KeyboardInterrupt is encountered but states current cylce to
                allow to resume the optimization.

        """
        ## initialize solver
        solve_s = steepest_descent(self.Egs,**self.config.iniSD_s)
        solve_u = steepest_descent(self.Egu,**self.config.iniSD_u)
        ## initialize parameters
        dc = 1
        raising = True
        ## capture keyboard interrupt
        try:
            cycle_ = int(cycle)#+int(self.config.map_mode<0)
            ## set starting values (1)
            self.s_ = field(self.z,val=self.config.s0,target=self.k)
            self.u_ = field(self.z,val=self.config.u0,target=self.k)
            self.S.set_power(self.config.p0,bare=True) ## p0 is bare(!)
            self.D_ = field(self.z,val=None,target=self.k)
            self.F_ = field(self.z,val=None,target=self.k)
            ## iterate
            while(True):
                ## cycle printout
                if(cycle_>=0):
                    self.note.cprint("\ncycle : %u"%cycle_)
                else:
                    self.note.cprint("\naftermath-cycle : %u"%cycle_.__neg__())
                ## raise s
                if(cycle_==0):
                    self._raise_s(cycle_,solve_s=solve_s)
                ## reset
                elif(cycle_==1)and(self.config.map_mode>0):
                    #self._io = np.array([[],[]],dtype=np.int)
                    self.u_.set_val(self.config.u0)
                ## set starting values (2)
                elif(cycle_==2):
                    self.D_.set_val(self.config.D0)
                    self.F_.set_val(self.config.F0)
                    ## reset restriction
                    self._restrict = False
                ## inference on point-like component(s)
                convergence,raising = self._infer_u(cycle_,solve_u=solve_u,raising=raising)
                ## reset
                if(cycle_==1)and(self.config.map_mode>0):
                    self.s_.set_val(self.config.s0)
                    self.S.set_power(self.config.p0,bare=True) ## p0 is bare(!)
                ## inference on diffuse component(s)
                convergence *= self._infer_s(cycle_,solve_s=solve_s)
                ## check convergence
                if(cycle_>=2)and(convergence):
                    if(self.config.aftermath):
                        cycle_ = -1
                        dc = -1
                    else:
                        break
                elif((cycle_<0)and(cycle_==self.config.aftermath))or((cycle_<0)and(self.config.aftermath>0)and(convergence)):
                    break
                else:
                    cycle_ += dc ## cylce on
        except(KeyboardInterrupt):
            raise KeyboardInterrupt(about._errors.cstring("KEYBOARDINTERRUPT in cycle %u."%cycle_)) ## resume?

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _save(self,final=False,char=None,cycle=None):
        if(final):
            np.save(os.path.join(self.config.wd,"s.npy"),self.s_)
            np.save(os.path.join(self.config.wd,"p.npy"),self.S.get_power())
            np.save(os.path.join(self.config.wd,"u.npy"),self.u_)
            np.save(os.path.join(self.config.wd,"D.npy"),self.D_)
            np.save(os.path.join(self.config.wd,"F.npy"),self.F_)
        if(self.config.saves)and(char is not None):
            if(cycle is None):
                raise ValueError(about._errors.cstring("ERROR: invalid call."))
            td = os.path.join(self.config.wd,"d3po_tmp")
            if(char=='s'):
                np.save(os.path.join(td,"s_%03u.npy"%cycle),self.s_)
            if(char=='p'):
                np.save(os.path.join(td,"p_%03u.npy"%cycle),self.S.get_power())
            if(char=='u'):
                np.save(os.path.join(td,"u_%03u.npy"%cycle),self.u_)
            if(char=='D'):
                if(cycle<0):
                    np.save(os.path.join(td,"Da%03u.npy"%cycle.__neg__()),self.D_)
                else:
                    np.save(os.path.join(td,"D_%03u.npy"%cycle),self.D_)
            if(char=='F'):
                if(cycle<0):
                    np.save(os.path.join(td,"Fa%03u.npy"%cycle.__neg__()),self.F_)
                else:
                    np.save(os.path.join(td,"F_%03u.npy"%cycle),self.F_)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _load_array(self,char,resume,tempfiles):
        ## select load
        if(isinstance(resume,bool)):
            ## load final save
            filename = os.path.join(self.config.wd,char,".npy")
            if(os.path.exists(filename)):
                self.note.cflush("\nloading : '"+filename+"'")
                return np.load(filename)
        if(isinstance(resume,bool))or(int(resume)<0):
            ## load last "tmp" save
            temp = [int(ff[3:][:-4]) for ff in tempfiles if(ff[:2]==char+'a')]
            if(len(temp)>0):
                last = max(temp)
                filename = os.path.join(self.config.wd,"d3po_tmp/%sa%03u.npy"%(char,last))
                if(os.path.exists(filename)):
                    self.note.cflush("\nloading : '"+filename+"'")
                    return np.load(filename)
            temp = [int(ff[3:][:-4]) for ff in tempfiles if(ff[:2]==char+'_')]
            if(len(temp)>0):
                last = max(temp)
                filename = os.path.join(self.config.wd,"d3po_tmp/%s_%03u.npy"%(char,last))
                if(os.path.exists(filename)):
                    self.note.cflush("\nloading : '"+filename+"'")
                    return np.load(filename)
        if(int(resume)>0):
            ## load specific "tmp" save
            filename = os.path.join(self.config.wd,"d3po_tmp/%s_%03u.npy"%(char,resume-1))
            if(os.path.exists(filename)):
                self.note.cflush("\nloading : '"+filename+"'")
                return np.load(filename)
        return None

    def _load(self,resume):
        ## check resume
        if(not resume):
            return 0
        ## get "tmp" file list
        td = os.path.join(self.config.wd,"d3po_tmp")
        if(os.path.exists(td)):
            tempfiles = [os.path.basename(ff) for ff in os.listdir(td)]
        else:
            tempfiles = [] ## empty
        ## load arrays
        array = self._load_array('s',resume,tempfiles)
        if(array is not None):
            self.config.s0 = array
        array = self._load_array('u',resume,tempfiles)
        if(array is not None):
            self.config.u0 = array
        array = self._load_array('p',resume,tempfiles)
        if(array is not None):
            self.config.p0 = array
            self.config.t0 = None
        array = self._load_array('D',resume,tempfiles)
        if(array is not None):
            self.config.D0 = array
        array = self._load_array('F',resume,tempfiles)
        if(array is not None):
            self.config.F0 = array
        self.note.cprint("\n... done.")
        ## return recycle
        if(isinstance(resume,bool)):
            return 2
        else:
            return resume

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get(self,s=False,p=False,t=False,u=False,D=False,F=False,rho=False,rho_s=False,rho_u=False,lmbda=False,lmbda_s=False,lmbda_u=False,s_err=False,u_err=False,regularize=None,weight=True,**kwargs):
        """
            Returns requested objects(s).

            The ordering of the returned objects follows the order of the
            parameters (see below).

            Parameters
            ----------
            s : bool
                Whether to return the diffuse signal field or not.
            p : bool
                Whether to return the power specturm or not.
            t : bool
                Whether to return the logarithmic power spectrum or not.
            u : bool
                Whether to return the point-like signal field or not.
            D : bool
                Whether to return the diagonal of the diffuse covariance or not.
            F : bool
                Whether to return the diagonal of the point-like covariance or not.
            rho : bool
                Whether to return the total photon flux or not.
            rho_s : bool
                Whether to return the diffuse photon flux or not.
            rho_u : bool
                Whether to return the point-like photon flux or not.
            lmbd : bool
                Whether to return the total reproduced photon counts or not.
            lmbd_s : bool
                Whether to return the diffuse reproduced photon counts or not.
            lmbd_u : bool
                Whether to return the point-like reproduced photon counts or not.
            s_err : bool
                Whether to return the relative error on the diffuse flux or not.
            u_err : bool
                Whether to return the relative error on the point-like flux or not.
            regularize : bool
                Whether to enforce regularization of the fluxes or not (default: None).
            weight : bool
                Whether to weight the fluxes or not (default: True).

            Note
            ----
            Probing errors (usually causing divergent or negative errors) are
            masked by zero valued entries (for numerical reasons) while the
            true value remains unknown.
            All fluxes should be weighted unless they represent data-like
            quantities. This (and other scalar factors) can be treated in the
            definition of the response.

            Retruns
            -------
            x : {field, array, tuple}
                Requested object or tuple thereof.

        """
        current = []
        if(s):
            current.append(self.s_)
        if(p):
            current.append(self.S.get_power())
        if(t):
            current.append(np.log(self.S.get_power()))
        if(u):
            current.append(self.u_)
        if(D):
            current.append(self.D_)
        if(F):
            current.append(self.F_)
        if(rho):
            if(self.config.map_mode>0):
                current.append(exp(self.s_eff()+exp(self.u_eff())).weight(power=weight))
            else:
                current.append(exp(self.s_eff()).weight(power=weight))
        if(rho_s):
            current.append(exp(self.s_eff(regularize=regularize)).weight(power=weight))
        if(rho_u):
            current.append(exp(self.u_eff(regularize=regularize)).weight(power=weight))
        if(lmbda):
            if(self.config.map_mode>0):
                current.append(self.R(exp(self.s_eff())+exp(self.u_eff())))
            else:
                current.append(self.R(exp(self.s_eff())))
        if(lmbda_s):
            current.append(self.R(exp(self.s_eff(regularize=regularize))))
        if(lmbda_u):
            current.append(self.R(exp(self.u_eff(regularize=regularize))))
        if(s_err):
            current.append(sqrt(exp(self.D_)-1))
        if(u_err):
            current.append(sqrt(exp(self.F_)-1))

        if(len(current)==0):
            return None
        elif(len(current)==1):
            return current[0]
        else:
            return tuple(current)

    def get_operators(self):
        """
            Returns all operators.

            Retruns
            -------
            X : tuple
                Tuple of operators: The diffuse prior covariance, the spectral
                projector, the likelihood part of the diffuse posterior
                covariance, the diffuse and point-like posterior covariance.

        """
        return self.S,self.Sk,self.M,self.D,self.F

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def solve(self,d,resume=False,**kwargs):
        """
            Solves the problem.

            Parameters
            ----------
            d : {list, array, field}
                A given data set.
            resume : {bool, integer}, *optional*
                Restarting cycle (default: False).

            See Also
            --------
            optimize

            Raises
            ------
            ValueError
                If data domain and response target do not match, or if negative
                count data is encountered.

        """
        if(self.config.map_mode<0):
            self.note.cprint('D3PO "light" version '+__version__)
        else:
            self.note.cprint("D3PO version "+__version__)

        ## check data
        if(not isinstance(d,field)):
            d = field(self.R.target,val=np.asarray(d),target=self.R.target)
        elif(d.domain!=self.R.target):
            raise ValueError(about._errors.cstring("ERROR: invalid domain."))
        if(np.any(d.val<0)):
            raise ValueError(about._errors.cstring("ERROR: negative counts."))
        elif(np.any(d.val!=np.round(d.val,0))):
            about.warnings.cprint("WARNING: non-integer data.")
        self.d = d

        ## fix random seed
        if(self.config.seed is not None):
            np.random.seed(self.config.seed)

        ## check resume
        recycle = self._load(resume)

        ## check if "light"
        if(self.config.map_mode<0):
            if(hasattr(self.R,"mask_out_pixels"))and(hasattr(self.R,"mask")):
                ## activate masking
                if(isinstance(self.R.mask_out_pixels,switch)):
                    self.R.mask_out_pixels.on()
                else:
                    self.R.mask_out_pixels = True

        ## optimize
        self.optimize(cycle=recycle)
        self.note.cprint("\nDONE.")

        ## save solution
        self._save(final=True)
        self.note.cprint("\nsave path : "+self.config.wd)

        ## return solution
        return self.get(**kwargs)

##=============================================================================



##-----------------------------------------------------------------------------

if(__name__=="__main__"):

    print("D3PO version "+__version__)
    print("\n**Command line execution not yet implemented**")
    print("\nusage:\t$ python\n\t>>> import d3po\n\t>>> p = d3po.problem([RESPONSE_OPERATOR],[CONFIG_FILE])\n\t>>> p.solve([DATA_ARRAY])")
    print("\ndemo:\t$ python demo.py\n")

##-----------------------------------------------------------------------------

