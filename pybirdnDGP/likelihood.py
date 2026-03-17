import os, sys
import numpy as np
from scipy.constants import c as c_light
from scipy.linalg import block_diag

from pybirdnDGP.correlator import Correlator
from pybirdnDGP.io_pb import ReadWrite
from pybirdnDGP.greenfunction import GreenFunction

class Likelihood(object):
    """EFT Likelihood"""
    def __init__(self, config, verbose=True):
        
        self.c = config
        self.io = ReadWrite()
        _, self.c_sky, self.d_sky = self.io.read(self.c, verbose=verbose) # skylists of formatted config dict and data dict
        self.nsky = len(self.c['sky'])
        self.set_data()
        self.set_config(verbose=verbose)
        self.set_class_settings()
        self.correlator_sky = [Correlator(self.c_sky[i]) for i in range(self.nsky)] # skylist of PyBird correlator engine
        self.set_eft_parameters()
        self.set_boost()
        self.zlist = np.array([i['z'] for i in self.c_sky])
        if self.c["with_bao_rec"]: self.alpha_sky = [None] * self.nsky # skylist of bao recon alpha 

    def set_data(self): 
        self.m_sky = [self.d_sky[i]['mask'] for i in range(self.nsky)] # mask for theory model
        self.y_sky = [self.d_sky[i]['y'] for i in range(self.nsky)]    # data to fit per sky
        self.p_sky = [self.d_sky[i]['p'] for i in range(self.nsky)]    # precision matrix per sky
        self.y_all = np.concatenate(self.y_sky)
        self.p_all = block_diag(*self.p_sky) 
        self.out = [{} for i in range(self.nsky)]
    
    def set_config(self, verbose=True):
        options = ['get_maxlkl', 'with_boss_correlated_skies_prior', 'with_rs_marg']
        if verbose: print ('-----------------------')
        for keys in options:
            if not keys in self.c: self.c[keys] = False
            if verbose: print (keys, ':', self.c[keys])
        if verbose: print ('-----------------------')

    def set_class_settings(self):
        log10kmax = 0
        if self.c["with_nnlo_counterterm"] or self.c["with_rs_marg"]: log10kmax = 1 # slower, but needed for the wiggle-no-wiggle split
        self.class_settings = {'output': 'mPk', 'z_max_pk': max([self.d_sky[i]["z"] for i in range(self.nsky)]), 'P_k_max_h/Mpc': 10.**log10kmax}
        self.kin = np.logspace(-5, log10kmax, 200)
        return

    def set_eft_parameters(self):
        self.b_name = [param for param in self.c["eft_prior"]]
        self.bg_name = [param for param, prior in self.c["eft_prior"].items() if prior["type"] == 'marg_gauss']
        self.bg_prior_mean = np.array([self.c["eft_prior"][param]["mean"] for param in self.bg_name])
        self.bg_prior_sigma = np.array([self.c["eft_prior"][param]["range"] for param in self.bg_name]) 
        self.bng_name = [param for param, prior in self.c["eft_prior"].items() if prior["type"] == 'gauss']
        self.bng_prior_mean = np.array([self.c["eft_prior"][param]["mean"] for param in self.bng_name])
        self.bng_prior_sigma = np.array([self.c["eft_prior"][param]["range"] for param in self.bng_name])
        self.bnlog_name = [param for param, prior in self.c["eft_prior"].items() if prior["type"] == 'lognormal']
        self.bnlog_prior_mean = np.array([self.c["eft_prior"][param]["mean"] for param in self.bnlog_name])
        self.bnlog_prior_sigma = np.array([self.c["eft_prior"][param]["range"] for param in self.bnlog_name])

        self.Ng = len(self.bg_name)
        if self.Ng > 0: self.marg_lkl = True
        else: self.marg_lkl = False

        if self.marg_lkl:
            self.F2_bg_prior_matrix = np.eye(self.Ng*self.nsky)
            self.prior_inv_corr_matrix = np.eye(self.nsky)
            self.F2_bg_prior_matrix /= np.concatenate(self.bg_prior_sigma.T)**2
            
            bg_prior_mean = np.concatenate(self.bg_prior_mean.T)
            self.F1_bg_prior_mean = np.einsum('a,ab->b', bg_prior_mean, self.F2_bg_prior_matrix) 
            self.chi2_bg_prior_mean = np.einsum('a,b,ab->', bg_prior_mean, bg_prior_mean, self.F2_bg_prior_matrix) 
        
        return

    def rescaling_rules(self,b_name,Aap,S):
        A_dict = {'b1':S*np.sqrt(Aap), 
                  'b2':S**2*np.sqrt(Aap), 
                  'b3':S**4*Aap, 
                  'b4':S**2*np.sqrt(Aap), 
                  'cct':S**2*Aap, 
                  'cr1':S**2*Aap, 
                  'cr2':S**2*Aap, 
                  'ce0':Aap, 
                  'ce1':Aap/self.fsat_sigma2_1_sky, 
                  'ce2':Aap/self.fsat_sigma2_1_sky}
        return A_dict[b_name]

    def rescale_eft_parameters(self,Aap,S):
        A_dict = {'b1':S*np.sqrt(Aap), 
                  'b2':S**2*np.sqrt(Aap), 
                  'b3':S**4*Aap, 
                  'b4':S**2*np.sqrt(Aap), 
                  'cct':S**2*Aap, 
                  'cr1':S**2*Aap, 
                  'cr2':S**2*Aap, 
                  'ce0':Aap, 
                  'ce1':Aap/self.fsat_sigma2_1_sky, 
                  'ce2':Aap/self.fsat_sigma2_1_sky}
        
        self.b_name = [param for param in self.c["eft_prior"]]
        self.bg_name = [param for param, prior in self.c["eft_prior"].items() if prior["type"] == 'marg_gauss']
        self.bg_prior_mean = np.array([self.c["eft_prior"][param]["mean"]/A_dict[param] for param in self.bg_name])
        self.bg_prior_sigma = np.array([self.c["eft_prior"][param]["range"]/A_dict[param] for param in self.bg_name]) 
        self.bng_name = [param for param, prior in self.c["eft_prior"].items() if prior["type"] == 'gauss']
        self.bng_prior_mean = np.array([self.c["eft_prior"][param]["mean"]/A_dict[param] for param in self.bng_name])
        self.bng_prior_sigma = np.array([self.c["eft_prior"][param]["range"]/A_dict[param] for param in self.bng_name])


        self.Ng = len(self.bg_name)
        if self.Ng > 0: self.marg_lkl = True
        else: self.marg_lkl = False

        if self.marg_lkl:
            self.F2_bg_prior_matrix = np.eye(self.Ng*self.nsky)
            self.prior_inv_corr_matrix = np.eye(self.nsky)
            self.F2_bg_prior_matrix /= np.concatenate(self.bg_prior_sigma.T)**2
            
            bg_prior_mean = np.concatenate(self.bg_prior_mean.T)
            self.F1_bg_prior_mean = np.einsum('a,ab->b', bg_prior_mean, self.F2_bg_prior_matrix) 
            self.chi2_bg_prior_mean = np.einsum('a,b,ab->', bg_prior_mean, bg_prior_mean, self.F2_bg_prior_matrix) 
        
        return

    def get_rescaling(self):
        self.S_sky = self.sigma8_sky/self.sigma8_fid_sky
        if self.with_ap:
            Aap_sky = (self.H_fid_sky/self.H_sky)*(self.DA_sky/self.DA_fid_sky)**2
            self.Aap_sky = 1/Aap_sky
        else:
            self.Aap_sky = np.ones_like(self.S_sky)

    def set_rescaling(self,fiducial):
        z  = self.zlist
        self.with_ap = self.c_sky[0]['with_ap']
            
        def Hubble(Om,z,w0=-1.,wa=0.): 
            return ((Om)*(1+z)**3.+(1-Om)*(1+z)**(3.*(1.+w0+wa))*np.exp(-3.*wa*z/(1.+z)))**0.5
            
        def DA(Om,z,w0=-1.,wa=0.):
            zz = np.linspace(1e-5, z, num=30, endpoint=True)
            return np.trapz(1/Hubble(Om,zz,w0,wa), x=zz) / (1+z)
            
        self.sigma8_fid_sky = fiducial.sigma_rz(8,z)
        self.fsat_sigma2_1_sky = np.array([i['fid']['fsat']*i['fid']['sigma1_eff']**2 for i in self.d_sky])
        if self.with_ap:
            self.H_fid_sky  = np.array([Hubble(fiducial.Omega0_m,i,w0=fiducial.w0_fld,wa=fiducial.wa_fld) for i in z])
            self.DA_fid_sky = np.array([DA(fiducial.Omega0_m,i,w0=fiducial.w0_fld,wa=fiducial.wa_fld) for i in z])

    def set_boost(self): 
        self.optipath_chi2 = np.einsum_path('a,b,ab->', self.y_all, self.y_all, self.p_all, optimize='optimal')[0]
        if self.marg_lkl: 
            dummy_ak = np.zeros(shape=(self.Ng, self.y_all.shape[0]))
            self.optipath_F2 = np.einsum_path('ak,bp,kp->ab', dummy_ak, dummy_ak, self.p_all, optimize='optimal')[0]
            self.optipath_F1 = np.einsum_path('ak,p,kp->a', dummy_ak, self.y_all, self.p_all, optimize='optimal')[0]
            self.optipath_bg = np.einsum_path('a,ab->b', self.y_all, self.p_all, optimize='optimal')[0] 
        return

    def get_chi2_marg(self, Tng_k, Tg_bk, P): 
        """Marginalized chi2"""
        F2 = np.einsum('ak,bp,kp->ab', Tg_bk, Tg_bk, P, optimize=self.optipath_F2) 
        F1 = np.einsum('ak,p,kp->a', Tg_bk, Tng_k, P, optimize=self.optipath_F1)
        F0 = self.get_chi2_non_marg(Tng_k, P) 

        F1 -= self.F1_bg_prior_mean
        F2 += self.F2_bg_prior_matrix
        invF2 = np.linalg.inv(F2)

        chi2 = F0 - np.einsum('a,b,ab->', F1, F1, invF2, optimize=self.optipath_chi2) + np.linalg.slogdet(F2)[1] 
        chi2 += self.chi2_bg_prior_mean
        bg = - np.einsum('a,ab->b', F1, invF2, optimize=self.optipath_bg) 

        return chi2, bg

    def get_chi2_non_marg(self, T_k, P):
        """Standard non-marginalized chi2"""
        chi2 = np.einsum('k,p,kp->', T_k, T_k, P, optimize=self.optipath_chi2) # 
        return chi2

    def get_prior(self, b_sky):
        """Prior"""
        def _get_prior(bs, prior_mean, prior_sigma, prior_inv_corr_mat=None, prior_type='gauss'):
            if prior_type == 'gauss': 
                prior = - 0.5 * np.einsum( 'n,nm,m->', bs - prior_mean, prior_sigma**-2 * prior_inv_corr_mat, bs - prior_mean )
            elif prior_type == 'lognormal':
                if any(b <= 0. for b in bs): prior = - 0.5 * np.inf
                else: prior = - 0.5 * np.einsum( 'n,nm,m->', np.log(bs) - prior_mean, prior_sigma**-2 * prior_inv_corr_mat, np.log(bs) - prior_mean ) #- np.sum( np.log(np.einsum('n,nm->m', bs, np.linalg.inv(prior_inv_corr_mat))), axis=0 ) #- np.sum(np.log(bs), axis=0)          
            elif prior_type == 'single_gauss': 
                prior = - 0.5 * (bs - prior_mean)**2 * prior_sigma**-2
            return prior
        prior = 0.
        for i, param in enumerate(self.bng_name): prior += _get_prior(np.array([b_sky[j][param] for j in range(self.nsky)]), self.bng_prior_mean[i], self.bng_prior_sigma[i], self.prior_inv_corr_matrix, prior_type='gauss')
        for i, param in enumerate(self.bnlog_name): prior += _get_prior(np.array([b_sky[j][param] for j in range(self.nsky)]), self.bnlog_prior_mean[i], self.bnlog_prior_sigma[i], self.prior_inv_corr_matrix, prior_type='lognormal')
        if self.marg_lkl and self.c["get_maxlkl"]: 
            for i, param in enumerate(self.bg_name): prior += _get_prior(np.array([b_sky[j][param] for j in range(self.nsky)]), self.bg_prior_mean[i], self.bg_prior_sigma[i], self.prior_inv_corr_matrix, prior_type='gauss')
        return prior

    def get_alpha_bao_rec(self, class_engine, i_sky=0):
        rd_by_rdfid = class_engine.rs_drag() / self.d_sky[i_sky]['bao_rec_fid']['rd']
        DM_by_DMfid = class_engine.angular_distance(self.d_sky[i_sky]['z']) / self.d_sky[i_sky]['bao_rec_fid']['D']
        H_by_Hfid = class_engine.Hubble(self.d_sky[i_sky]['z']) * c_light*1e-3 / self.d_sky[i_sky]['bao_rec_fid']['H']
        alpha_par = 1. / (rd_by_rdfid * H_by_Hfid)
        alpha_per = DM_by_DMfid / rd_by_rdfid
        return alpha_par, alpha_per
    
    def set_bao_rec(self, alphas, Tng_k, Tg_bk=None):
        Tng_k = np.concatenate((Tng_k, alphas))
        if Tg_bk is not None: Tg_bk = np.pad(Tg_bk, [(0, 0), (0, 2)], mode='constant', constant_values=0)
        return Tng_k, Tg_bk

    def set_cosmoprimo(self,M,Omega0_rc,rescale_pklin=False):
        fo = M.get_fourier()
        pk = fo.pk_interpolator(of='delta_cb')
        z  = self.zlist

        def get_f(z):
            sigma8  = fo.sigma8_z(z, of='delta_cb') #"Why not delta_m?"
            fsigma8 = fo.sigma8_z(z, of='theta_cb')
            f       = fsigma8 / sigma8
            return f
            
        def Hubble(Om,z,w0=-1.,wa=0.): 
            return ((Om)*(1+z)**3.+(1-Om)*(1+z)**(3.*(1.+w0+wa))*np.exp(-3.*wa*z/(1.+z)))**0.5
            
        def DA(Om,z,w0=-1.,wa=0.):
            zz = np.linspace(1e-5, z, num=30, endpoint=True)
            return np.trapz(1/Hubble(Om,zz,w0,wa), x=zz) / (1+z)
            

        log10kmax = 0
        if self.c["with_nnlo_counterterm"]: log10kmax = 1

        if rescale_pklin:
            GF_GR = GreenFunction(M.Omega0_m, Omega0_rc=None,w0=M.w0_fld,wa=M.wa_fld,H0=M.h*100)
            GF_MG = GreenFunction(M.Omega0_m, Omega0_rc=Omega0_rc,w0=M.w0_fld,wa=M.wa_fld,H0=M.h*100)
            x = np.log(1/(1+z))
            rescale_factor = (GF_MG.D(x)/GF_GR.D(x))
            self.f_sky     = GF_MG.fplus(x)
        else:
            rescale_factor = np.ones_like(z)
            self.f_sky     = get_f(z)
        self.kk         = np.logspace(-5, log10kmax, 200)  # k in h/Mpc
        self.pklin_sky  = rescale_factor**2*pk(self.kk, z=z).T
        self.sigma8_sky = rescale_factor*M.sigma_rz(8,z)
        if self.with_ap:
            self.H_sky  = np.array([Hubble(M.Omega0_m,i,w0=M.w0_fld,wa=M.wa_fld) for i in z])
            self.DA_sky = np.array([DA(M.Omega0_m,i,w0=M.w0_fld,wa=M.wa_fld) for i in z])
        
        
    def loglkl_cosmoprimo(self, free_b, free_b_name, exact_time_params, need_cosmo_update=True): 
        ###Should call self.set_cosmoprimo() and self.set_rescaling(fiducial) and self.get_rescaling() before
        self.rescale_eft_parameters(self.Aap_sky,self.S_sky)
        free_b_sky = np.array(free_b).reshape(self.nsky, -1)
        free_b_name_sky = np.array(free_b_name).reshape(self.nsky, -1)
        b_sky = [] # skylist of EFT parameters dict
        for i in range(self.nsky):
            b_sky.append({bn: free_b_sky[i][n] for n, fbn in enumerate(free_b_name_sky[i]) for bn in self.b_name if fbn.split('_', 1)[0] == bn})
            b_sky[i].update({bn: 0. for bn in self.b_name if bn not in b_sky[i]})

        if need_cosmo_update:
            for i in range(self.nsky): 
                if self.with_ap:
                    cosmo_dict_local = {'kk': np.copy(self.kk), 'pk_lin': np.copy(self.pklin_sky[i]),'f':self.f_sky[i],
                                        'H':self.H_sky[i], 'DA':self.DA_sky[i]}
                else:
                    cosmo_dict_local = {'kk': np.copy(self.kk), 'pk_lin': np.copy(self.pklin_sky[i]),'f':self.f_sky[i]}
                self.correlator_sky[i].compute(cosmo_dict_local | exact_time_params)
        
        if self.marg_lkl:
            Tng_k, Tg_bk = [], []
            for i in range(self.nsky):
                Tng_k.append( self.correlator_sky[i].get(b_sky[i]).reshape(-1)[self.m_sky[i]] )
                Tg_bk.append( self.correlator_sky[i].getmarg(b_sky[i], self.bg_name)[:, self.m_sky[i]] )
                if self.c["with_bao_rec"]: Tng_k[i], Tg_bk[i] = self.set_bao_rec(self.alpha_sky[i], Tng_k[i], Tg_bk[i])
            chi2, bg = self.get_chi2_marg(np.concatenate(Tng_k)-self.y_all, block_diag(*Tg_bk), self.p_all) 
            if self.c["get_maxlkl"]: 
                bg_resh = bg.reshape(self.nsky,-1)
                for i in range(self.nsky): b_sky[i].update({p: b for p, b in zip(self.bg_name, bg_resh[i])})

        if not self.marg_lkl or self.c["get_maxlkl"]: 
            chi2 = 0.
            for i in range(self.nsky):
                Tng_k = self.correlator_sky[i].get(b_sky[i]).reshape(-1)[self.m_sky[i]]
                if self.c["with_bao_rec"]: Tng_k, _ = self.set_bao_rec(self.alpha_sky[i], Tng_k, None)
                chi2_i = self.get_chi2_non_marg(Tng_k-self.y_sky[i], self.p_sky[i]) 
                if self.c["write"]["fake"] or self.c["write"]["save"]: self.set_out(self.correlator_sky[i].get(b_sky[i]), chi2_i, b_sky[i], M=cosmo_engine, i_sky=i)
                chi2 += chi2_i

        prior = self.get_prior(b_sky)
        lkl = - 0.5 * chi2 + prior
        return lkl

    def loglkl(self, free_b, free_b_name, cosmo_engine, cosmo_module='class', need_cosmo_update=True): 

        pad = 0
        if self.c["with_rs_marg"]: 
            alpha_rs_marg = free_b[free_b_name.index("alpha_rs")]
            pad += 1 
        free_b_sky = np.array(free_b[pad:]).reshape(self.nsky, -1)
        free_b_name_sky = np.array(free_b_name[pad:]).reshape(self.nsky, -1)
        b_sky = [] # skylist of EFT parameters dict
        for i in range(self.nsky):
            b_sky.append({bn: free_b_sky[i][n] for n, fbn in enumerate(free_b_name_sky[i]) for bn in self.b_name if fbn.split('_', 1)[0] == bn})
            b_sky[i].update({bn: 0. for bn in self.b_name if bn not in b_sky[i]})

        if need_cosmo_update:
            for i in range(self.nsky): 
                self.correlator_sky[i].compute(cosmo_dict=None, cosmo_module=cosmo_module, cosmo_engine=cosmo_engine)
                if self.c["with_bao_rec"]: self.alpha_sky[i] = self.get_alpha_bao_rec(cosmo_engine, i_sky=i)
        
        if self.marg_lkl:
            Tng_k, Tg_bk = [], []
            for i in range(self.nsky):
                Tng_k.append( self.correlator_sky[i].get(b_sky[i]).reshape(-1)[self.m_sky[i]] )
                Tg_bk.append( self.correlator_sky[i].getmarg(b_sky[i], self.bg_name)[:, self.m_sky[i]] )
                if self.c["with_bao_rec"]: Tng_k[i], Tg_bk[i] = self.set_bao_rec(self.alpha_sky[i], Tng_k[i], Tg_bk[i])
            chi2, bg = self.get_chi2_marg(np.concatenate(Tng_k)-self.y_all, block_diag(*Tg_bk), self.p_all) 
            if self.c["get_maxlkl"]: 
                bg_resh = bg.reshape(self.nsky,-1)
                for i in range(self.nsky): b_sky[i].update({p: b for p, b in zip(self.bg_name, bg_resh[i])})

        if not self.marg_lkl or self.c["get_maxlkl"]: 
            chi2 = 0.
            for i in range(self.nsky):
                Tng_k = self.correlator_sky[i].get(b_sky[i]).reshape(-1)[self.m_sky[i]]
                if self.c["with_bao_rec"]: Tng_k, _ = self.set_bao_rec(self.alpha_sky[i], Tng_k, None)
                chi2_i = self.get_chi2_non_marg(Tng_k-self.y_sky[i], self.p_sky[i]) 
                if self.c["write"]["fake"] or self.c["write"]["save"]: self.set_out(self.correlator_sky[i].get(b_sky[i]), chi2_i, b_sky[i], M=cosmo_engine, i_sky=i)
                chi2 += chi2_i

        prior = self.get_prior(b_sky)
        lkl = - 0.5 * chi2 + prior
        return lkl
    
    def set_out(self, y_arr, chi2, eft_parameters, M=None, i_sky=0): 
        xmask = self.d_sky[i_sky]['mask_arr']
        self.out[i_sky]['y_arr'] = [y_arr[i, xmask_i] for i, xmask_i in enumerate(xmask)]
        self.out[i_sky]['y'] = y_arr.reshape(-1)[self.m_sky[i_sky]]
        self.out[i_sky]['x_unmasked'] = self.d_sky[i_sky]['x']
        self.out[i_sky]['y_arr_unmasked'] = y_arr
        if self.c["with_bao_rec"]: self.out[i_sky]['alpha'] = self.alpha_sky[i_sky]
        self.out[i_sky]['chi2'] = chi2
        self.out[i_sky]['eft_parameters'] = eft_parameters
        if M is not None: # class engine
            self.out[i_sky]['cosmo'] = {'omega_b': M.omega_b(), 'omega_cdm': M.Omega0_cdm() * M.h()**2, 'Omega_k': M.Omega0_k(), 'Omega_nu': M.Omega_nu} 
            self.out[i_sky]['cosmo'].update(M.get_current_derived_parameters(["Omega_m", "h", "A_s", "n_s", "sigma8"]))
    
    def write(self): 
        self.io.write(self.c, self.d_sky, self.out)
