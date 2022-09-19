import numpy as np
#from numpy.core.arrayprint import array_repr
#from numpy.core.fromnumeric import shape
#from numpy.lib.function_base import append
#from numpy.lib.shape_base import dstack
from scipy import linalg as lg
from scipy import signal as sg
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.signal import convolve2d

#from scipy.ndimage.measurements import center_of_mass

import time

class LIFT:
    def __init__(self, tel, modeBasis, astigmatism_OPD, iterations):
        self.tel             = tel
        self.modeBasis       = modeBasis
        self.astigmatism_OPD = astigmatism_OPD
        self.iterations      = iterations
        self.oversampling    = 1

    # Since coefficient vector A can contain Nones, they must be ignored during the calculation
    def pack(self, A_vec):
        return np.array( [a_k for a_k in A_vec if (a_k != None) and (not np.isnan(a_k))] )

    # Back to the form where position of the coeeficient in the vector correspond to the number of  mode
    def unpack(self, coef_vec, mode_nums):
        A_vec = [np.nan for i in range(max(mode_nums)+1)]
        for i, mode in enumerate(mode_nums): 
            A_vec[mode] = coef_vec[i]
        return np.array(A_vec)

    def print_modes(self, A_vec):
        for i in range(A_vec.shape[0]):
            val = A_vec[i]
            if (val != None) and (not np.isnan(val)):
                val = np.round(val, 4)
            print('Mode #', i, val)


    def generateLIFTinteractionMatrices(self, coefs, numerical=False):
        if isinstance(coefs, list):
            coefs = np.array(coefs)

        oversampling = 1
        initial_OPD = self.modeBasis.wavefrontFromModes(self.tel, coefs) + self.astigmatism_OPD
        
        H = []

        if not numerical:
            for point in self.tel.src.spectrum:
                wavelength = point['wavelength']
                initial_amplitude = np.sqrt(self.tel.flux(point['flux'], self.tel.det.sampling_time)) * self.tel.pupil

                initial_phase = 2 * np.pi / wavelength * initial_OPD
                Pd = np.conj( self.tel.PropagateField(initial_amplitude, initial_phase, wavelength, return_intensity=False, oversampling=self.oversampling) )
                
                H_spectral = []
                for i in range(coefs.shape[0]):
                    if (coefs[i] is not None) and (not np.isnan(coefs[i])):
                        buf = self.tel.PropagateField(self.modeBasis.modesFullRes[:,:,i]*initial_amplitude, initial_phase, wavelength, return_intensity=False, oversampling=self.oversampling)
                        derivative = 2*np.real(1j * buf * Pd) * 2 * np.pi / wavelength

                        if self.tel.object is not None:
                            derivative = sg.convolve2d(derivative, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()
                        H_spectral.append( derivative.reshape([buf.shape[0]*buf.shape[1]]) )
                H.append(np.dstack(H_spectral)[0])
        else: 
            delta = 1e-9 # [nm]
            for point in self.tel.src.spectrum:
                H_spectral = []
                for i in range(coefs.shape[0]):
                    if (coefs[i] is not None) and (not np.isnan(coefs[i])):       
                        self.tel.src.OPD =  (self.modeBasis.modesFullRes[:,:,i] * delta) + initial_OPD
                        tmp1 = self.tel.ComputePSF()                

                        self.tel.src.OPD = -(self.modeBasis.modesFullRes[:,:,i] * delta) + initial_OPD
                        tmp2 = self.tel.ComputePSF()
                            
                        derivative = (tmp1 - tmp2) / 2 / delta
                            
                        if self.tel.object is not None:
                            derivative = sg.convolve2d(derivative, self.tel.object, boundary='symm', mode='same')
                        H_spectral.append( derivative.reshape([tmp2.shape[0]*tmp2.shape[1]]) )
                H.append(np.dstack(H_spectral)[0])
                
        return np.dstack(H).sum(axis=2) # sum all spectral interaction matricies


    def generateLIFTinteractionMatricesGPU(self, coefs, numerical=False):
        if isinstance(coefs, list):
            coefs = np.array(coefs)

        oversampling = 1
        initial_OPD = self.modeBasis.wavefrontFromModes(self.tel, coefs) + self.astigmatism_OPD
        
        H = []

        if not numerical:
            for point in self.tel.src.spectrum:
                wavelength = point['wavelength']
                initial_amplitude = np.sqrt(self.tel.flux(point['flux'], self.tel.det.sampling_time)) * self.tel.pupil

                initial_phase = 2 * np.pi / wavelength * initial_OPD
                Pd = np.conj( self.tel.PropagateFieldGPU(initial_amplitude, initial_phase, wavelength, return_intensity=False, oversampling=self.oversampling) )
                
                H_spectral = []
                for i in range(coefs.shape[0]):
                    if (coefs[i] is not None) and (not np.isnan(coefs[i])):
                        buf = self.tel.PropagateFieldGPU(self.modeBasis.modesFullRes[:,:,i]*initial_amplitude, initial_phase, wavelength, return_intensity=False, oversampling=self.oversampling)
                        derivative = 2*np.real(1j * buf * Pd) * 2 * np.pi / wavelength

                        if self.tel.object is not None:
                            derivative = sg.convolve2d(derivative, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()
                        H_spectral.append( derivative.reshape([buf.shape[0]*buf.shape[1]]) )
                H.append(np.dstack(H_spectral)[0])
        else: #TODO: not yet checked!
            delta = 1e-9 # [nm]
            for point in self.tel.src.spectrum:
                H_spectral = []
                for i in range(coefs.shape[0]):
                    if (coefs[i] is not None) and (not np.isnan(coefs[i])):       
                        self.tel.src.OPD =  (self.modeBasis.modesFullRes[:,:,i] * delta) + initial_OPD
                        tmp1 = self.tel.ComputePSF(GPU=True)                

                        self.tel.src.OPD = -(self.modeBasis.modesFullRes[:,:,i] * delta) + initial_OPD
                        tmp2 = self.tel.ComputePSF(GPU=True)
                            
                        derivative = (tmp1 - tmp2) / 2 / delta
                            
                        if self.tel.object is not None:
                            derivative = sg.convolve2d(derivative, self.tel.object, boundary='symm', mode='same')
                        H_spectral.append( derivative.reshape([tmp2.shape[0]*tmp2.shape[1]]) )
                H.append(np.dstack(H_spectral)[0])
                
        return cp.array( np.dstack(H).sum(axis=2) ) # sum all spectral interaction matricies


    def Reconstruct(self, PSF, R_n, mode_ids, A_0=None, A_ref=None, verbous=False, optimize_norm=False):
        C  = []         # optimization criterion
        Hs = []         # interaction matrices for every iteration
        P_MLs = []      # estimators for every iteration
        A_ests = []     # history of estimated coefficients
        
        mode_ids.sort()

        # If coefficients ground truth is known
        if A_ref is not None:
            if np.all(np.abs(self.pack(A_ref)) < 1e-9):
                print('Warning: nothing to optimize, all ref. coefs. = 0')
                return A_ref, None, None

        # Account for the initial approximation for aberrations
        initial_OPD = 0.0
        A_est = np.zeros(max(mode_ids)+1)

        if A_0 is not None:
            modes_0 = np.where(np.isfinite(A_0))[0].tolist() # modes defined in the initial vector
            for i in set(mode_ids) & set(modes_0):
                A_est[i] = A_0[i]
            initial_OPD = self.modeBasis.wavefrontFromModes(self.tel, A_0)

        for i in set(range(max(mode_ids)+1)) - set(mode_ids):
            A_est[i] = np.nan
        A_est = self.pack(A_est)

        self.tel.src.OPD = self.astigmatism_OPD + initial_OPD
        PSF_0 = self.tel.ComputePSF()
    
        # Take a priory object information into account
        if self.tel.object is not None:
            PSF_0 = sg.convolve2d(PSF_0, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()

        A_ests.append(np.copy(A_est)) # saving coefficients history
        PSF_cap = np.copy(PSF_0)      # saving PSF history
        inv_R_n = 1.0 / np.clip(R_n.reshape(R_n.shape[0]*R_n.shape[1]), a_min=1e-6, a_max=R_n.max())

        criterion  = lambda i: np.abs(C[i]-C[i-1]) / C[i]
        coefs_norm = lambda v: np.linalg.norm(v, ord=2)

        for i in range(self.iterations):
            PSF_buf = np.copy(PSF)
            inv_R_n_buf = np.copy(inv_R_n)

            if optimize_norm:
                PSF_buf *= PSF_cap.max() / PSF.max()
                inv_R_n_buf /= PSF_cap.max() / PSF.max()

            dI = PSF_buf - PSF_cap
            dI = dI.flatten().astype('float64')

            # Check convergence
            C.append( np.dot(dI * inv_R_n, dI) )
            
            if i > 0 and (criterion(i) < 1e-6 or coefs_norm(A_ests[i]-A_ests[i-1]) < 1e-12):
                print('Criterion', criterion(i), 'is reached at iter.', i)
                break

            if A_ref is not None:
                if coefs_norm(self.pack(A_ref)-A_est) < 1e-12:
                    print('Err. vec. norm', coefs_norm(self.pack(A_ref)-A_est), 'is reached at iter.', i)
                    break
            
            # Generate interaction matricies
            H = self.generateLIFTinteractionMatrices(coefs=self.unpack(A_est, mode_ids))                               

            # Maximum likelyhood estimation
            P_ML = lg.pinv(H.T * inv_R_n @ H) @ H.T * inv_R_n
            d_A = P_ML @ dI
            A_est += d_A
            
            Hs.append(H)
            P_MLs.append(P_ML)
            A_ests.append(np.copy(A_est))

            if verbous:
                print( 'Iteration:', i )  
                if A_ref is not None:
                    print( 'Criterion:', criterion(i), ', err. vec. norm:', coefs_norm(self.pack(A_ref)*1e9-A_est*1e9))
                else:
                    print( 'Criterion:', criterion(i))
                self.print_modes(self.unpack(d_A*1e9, mode_ids))
                print()
            
            # Recreate the spot with reconstructed coefficients
            estimated_OPD = self.modeBasis.wavefrontFromModes(self.tel, self.unpack(A_est, mode_ids))
            self.tel.src.OPD = estimated_OPD + self.astigmatism_OPD
            PSF_cap = self.tel.ComputePSF()

            if self.tel.object is not None:
                PSF_cap = sg.convolve2d(PSF_cap, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()

        if optimize_norm:
            PSF_cap = PSF_cap * PSF.max() / PSF_cap.max()

        history = { #save intermediate data every iteration
                'P_ML' : np.dstack(P_MLs),
                'H'    : np.dstack(Hs),
                'A_est': np.squeeze( np.dstack(A_ests), axis=0 ),
                'C'    : np.array(C)
            }
        return self.unpack(A_est, mode_ids), PSF_cap, history

        
    def ReconstructGPU(self, PSF, R_n, mode_ids, A_0=None, PSF_0=None, A_ref=None, verbous=False, SR=1.0, optimize_norm=True):
        C  = []         # stopping criterion
        Hs = []         # interaction matrices for every iteration
        P_MLs = []      # estimators for every iteration
        A_ests = []     # history of A_est estimations
        mode_ids.sort() # assume that A vector is sorted as well

        if A_ref is not None:
            if np.all(np.abs(self.pack(A_ref)) < 1e-9):
                return self.unpack(A_ref, mode_ids), None, None

        # Account for the initial values of aberrations
        if A_0 is None:
            A_est = cp.zeros(self.modeBasis.modesFullRes.shape[2], dtype=cp.float32)
        else:
            A_est = cp.array(A_0, dtype=cp.float32)

        # Account for initial image shape
        if PSF_0 is None:
            if A_0 is None:
                self.tel.src.OPD = self.astigmatism_OPD
            else:
                initial_OPD = self.modeBasis.wavefrontFromModes(self.tel, A_0)
                self.tel.src.OPD = initial_OPD + self.astigmatism_OPD
            PSF_0 = self.tel.ComputePSF(GPU=True)
    
            # Take a prior object's shape knowledge into account
            if self.tel.object is not None:
                PSF_0 = sg.convolve2d(PSF_0, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()

        assert (PSF_0.shape[0]==PSF.shape[0]) and (PSF_0.shape[1]==PSF.shape[1]), 'Error: dimensions of generated PSF mismatch with the input PSF.'

        A_ests.append(cp.copy(A_est))
        PSF_cap = cp.array(PSF_0, dtype=cp.float32)

        R_n = cp.array(R_n)
        inv_R_n = 1.0 / cp.clip(R_n.reshape(R_n.shape[0]*R_n.shape[1]), a_min=1e-6, a_max=R_n.max())

        PSF_GPU = cp.array(PSF, dtype=cp.float32)

        #norm_factor = 1.0
        #norm_factors = [norm_factor]

        #if optimize_norm: # normalize input PSF to the same peak value as for synthetic
        #    norm_factor = SR*PSF_cap.sum() / PSF.sum()
        #    PSF_buf *= norm_factor
        #norm_factors.append(norm_factor)

        for i in range(self.iterations):
            PSF_buf = cp.copy(PSF_GPU)
    
            if optimize_norm: # normalize input PSF to the same peak value as for synthetic
                PSF_buf *= PSF_cap.max() / PSF_GPU.max()

            dI = PSF_buf - PSF_cap
            dI = dI.reshape(dI.shape[0] * dI.shape[1]).astype(cp.float32)

            # Check convergence
            C.append( cp.asnumpy( cp.dot(dI * inv_R_n, dI) ) )
            
            if i > 0 and ( np.abs(C[i]-C[i-1])/C[i] < 1e-6 or cp.linalg.norm(A_ests[i]-A_ests[i-1], ord=2) < 1e-12 ):
                print('Criterion', np.abs(C[i]-C[i-1]) / C[i], 'is reached at iter.', i)
                break

            if A_ref is not None:
                if cp.linalg.norm(cp.array(self.pack(A_ref))-A_est, ord=2) < 1e-12:
                    print('Err. vec. norm', np.linalg.norm(cp.array(self.pack(A_ref)-A_est), ord=2), 'is reached at iter.', i)
                    break
            
            # Generate interaction matricies
            #H = self.generateLIFTinteractionMatricesGPU(coefs=self.unpack(A_est.get(), mode_ids))                               
            H = self.generateLIFTinteractionMatricesGPU(coefs=self.unpack(A_est[mode_ids].get(), mode_ids))                               

            # Maximum likelyhood estimation
            P_ML = cp.linalg.pinv(H.T * inv_R_n @ H) @ H.T * inv_R_n
            d_A = P_ML @ dI
            A_est[mode_ids] += d_A
            
            Hs.append(H.get())
            P_MLs.append(P_ML.get())
            A_ests.append(cp.copy(A_est))

            if verbous:
                print( 'Iteration:', i )  
                if A_ref is not None:
                    print( 'Criterion:', np.abs(C[i]-C[i-1]) / C[i], ', err. vec. norm:', np.linalg.norm(self.pack(A_ref)*1e9-A_est*1e9, ord=2) )
                else:
                    print( 'Criterion:', np.abs(C[i]-C[i-1]) / C[i])
                self.print_modes(self.unpack(d_A*1e9, mode_ids))
                print()
            
            # Recreate the spot with reconstructed coefficients
            #estimated_OPD = self.modeBasis.wavefrontFromModes(self.tel, self.unpack(A_est.get(), mode_ids))
            estimated_OPD = self.modeBasis.wavefrontFromModes(self.tel, A_est.get())
            self.tel.src.OPD = estimated_OPD + self.astigmatism_OPD
            PSF_cap = cp.array( self.tel.ComputePSF(GPU=True) )
            if self.tel.object is not None:
                PSF_cap = convolve2d(PSF_cap, cp.array(self.tel.object), boundary='symm', mode='same') / self.tel.object.sum()

        if optimize_norm:
            PSF_cap = PSF_cap * PSF.max() / PSF_cap.max()
            #PSF_cap / norm_factor

        history = { #save intermediate data every iteration
                'P_ML' : np.dstack(P_MLs),
                'H'    : np.dstack(Hs),
                'A_est': cp.squeeze( cp.dstack(A_ests), axis=0 ).get(),
                'C'    : np.array(C)
            }

        #return cp.asnumpy( self.unpack(A_est.get(), mode_ids) ), PSF_cap.get(), history
        return A_est.get(), PSF_cap.get(), history



        
