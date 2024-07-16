import numpy as np
from scipy import linalg as lg
from scipy import signal as sg

try:
    import cupy as cp
    from cupyx.scipy import signal as csg
    global_gpu_flag = True

except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp  = np
    csg = sg
    global_gpu_flag = False

from OOPAO.tools.tools import bin_ndarray

from Rafael_tools.Rafael_tools import extractor,Propagate_field

class LIFT_OOPAO:
    def __init__(self, telescope, modeBasis,zeroPaddingFactor, oversampling, img_resolution, diversity_OPD, iterations, readoutnoise, object = None):
        global global_gpu_flag
        self.tel                 = telescope
        self.Z                   = modeBasis
        self.zeroPaddingFactor   = zeroPaddingFactor
        self.oversampling        = oversampling
        self.img_resolution      = img_resolution
        self.diversity_OPD       = diversity_OPD
        self.iterations          = iterations
        self.readoutnoise        = readoutnoise
        self.object              = object


    def print_modes(self, A_vec):
        for i in range(A_vec.shape[0]):
            val = A_vec[i]
            if val != None and not np.isnan(val): val = np.round(val, 4)
            print('Mode #', i, val)


    def obj_convolve(self, mat):
        xg = csg
        if self.object is not None:
            return xg.convolve2d(mat, self.object, boundary='symm', mode='same') / self.object.sum()
        else: return mat


    def generateLIFTinteractionMatrices(self, coefs, modes_ids, flux_norm=1.0, numerical=False):
        
        coefs = np.array(coefs,dtype=float)
        aux = coefs.copy() 
        aux[3] += self.diversity_OPD # adding phase diversity
        initial_OPD = np.squeeze(self.Z.modesFullRes@aux)
        
        H = []
        if not numerical:
            
            wavelength = self.tel.src.wavelength # m
            initial_amplitude = np.sqrt(self.tel.src.fluxMap*flux_norm)
            k = 2*np.pi/wavelength

            initial_phase = k * initial_OPD
            Pd = np.conj(extractor(Propagate_field(self.zeroPaddingFactor,initial_amplitude,initial_phase,self.tel),self.oversampling*self.img_resolution))
            
            H_spectral = []
            for i in modes_ids:
                buf = Propagate_field(self.zeroPaddingFactor,np.squeeze(self.Z.modesFullRes[:,:,i])*initial_amplitude,initial_phase,self.tel)
                buf1 = extractor(buf,self.oversampling*self.img_resolution)
                derivative = 2*bin_ndarray(np.real(1j*buf1*Pd), new_shape=(self.img_resolution,self.img_resolution), operation='sum')*k
                derivative = self.obj_convolve(derivative)

                H_spectral.append(derivative.flatten())   
            H.append(np.vstack(H_spectral).T)
        else:
            delta = 1e-9 # [nm]
            H_spectral = []
            for i in modes_ids:
                self.tel.OPD = (np.squeeze(self.Z.modesFullRes[:,:,i]) * delta) + initial_OPD
                self.tel.computePSF(self.zeroPaddingFactor)
                PSF_cropped = extractor(self.tel.PSF,self.img_resolution*self.oversampling)
                tmp1 = bin_ndarray(PSF_cropped,(self.img_resolution,self.img_resolution),'sum')*flux_norm

                self.tel.OPD = -(np.squeeze(self.Z.modesFullRes[:,:,i]) * delta) + initial_OPD
                self.tel.computePSF(self.zeroPaddingFactor)
                PSF_cropped = extractor(self.tel.PSF,self.img_resolution*self.oversampling)
                tmp2 = bin_ndarray(PSF_cropped,(self.img_resolution,self.img_resolution),'sum')*flux_norm

                derivative = self.obj_convolve((tmp1-tmp2)/2/delta)

                H_spectral.append( derivative.flatten() )
            H.append(np.vstack(H_spectral).T)
        
        return np.dstack(H).sum(axis=2) # sum all spectral interaction matricies


    def Reconstruct(self, PSF_inp, R_n, mode_ids, A_0=None, verbous=False, optimize_norm='sum'):
        """
        Function to reconstruct modal coefficients from the input PSF image using LIFT

        Parameters:
            PSF (ndarray):                   2-d array of the input PSF image to reconstruct.

            R_n (ndarray or string or None): The pixel weighting matrix for LIFT. It can be passed to the function.
                                             from outside, modeled ('model'), updated dynamically ('iterative'), or
                                             assumed to be just detector's readout noise ('None').
            mode_ids (ndarray or list):      IDs of the modal coefficients to be reconstructed
            A_0 (ndarray):                   initial assumtion for the coefficient values. In some sense, it acts as an additional 
                                             phase diversity on top of the main phase diversity which is passed when class is initialized.

            A_ref (ndarray):                 Reference coefficients to compare the reconstruction with. Useful only when ground-truth (A_ref) is known.

            verbous (bool):                  Set 'True' to print the intermediate reconstruction results.

            optimize_norm (string or None):  Recomputes the flux of the recontructed PSF iteratively. If 'None', the flux is not recomputed, 
                                             this is recommended only if the target brightness is precisely known. In mosyt of the case it is 
                                             recommended to switch it on. When 'sum', the reconstructed PSF is normalized to the sum of the pixels 
                                             of the input PSF. If 'max', then the reconstructed PSF is normalized to the maximal value of the input PSF.
        """

        convert = lambda x: x

        def PSF_from_coefs(coefs):
            aux = coefs.copy()
            aux[3] += self.diversity_OPD # adding phase diversity
            self.tel.OPD = np.squeeze(self.Z.modesFullRes@aux)
            self.tel.computePSF(self.zeroPaddingFactor)
            PSF_cropped = extractor(self.tel.PSF,self.img_resolution*self.oversampling)
            PSF_binned_cropped = bin_ndarray(PSF_cropped,(self.img_resolution,self.img_resolution),'sum')
        
            return self.obj_convolve(PSF_binned_cropped)
                   
        C      = []  # optimization criterion
        Hs     = []  # interaction matrices for every iteration
        P_MLs  = []  # estimators for every iteration
        A_ests = []  # history of estimated coefficients
        
        PSF = np.array(PSF_inp, dtype=np.float32)
        modes = np.sort( np.array(mode_ids, dtype=np.int32) )

        # Account for the initial assumption for the coefficients' values
        if A_0 is None:
            A_est = np.zeros(modes.max().item()+1, dtype=np.float32)
        else:
            A_est = np.array(A_0, dtype=np.float32)
        A_ests.append(np.copy(A_est))

        def normalize_PSF(PSF_in):
            if optimize_norm is not None and optimize_norm is not False:
                if optimize_norm == 'max': return (PSF_in/PSF_in.max(), PSF_in.max())
                if optimize_norm == 'sum': return (PSF_in/PSF_in.sum(), PSF_in.sum())
            else: return (PSF_in, 1.0)
        
        PSF_0, flux_cap = normalize_PSF(PSF_from_coefs(A_est)) # initial PSF assumtion, normalized to 1.0

        flux_scale = 1.0
        if optimize_norm is not None and optimize_norm is not False:
            if   optimize_norm == 'max': flux_scale = PSF.max()
            elif optimize_norm == 'sum': flux_scale = PSF.sum()
        
        PSF_cap = np.copy(PSF_0) * flux_scale

        criterion  = lambda i: np.abs(C[i]-C[i-1]) / C[i]
        coefs_norm = lambda v: np.linalg.norm(v[modes], ord=2)

        def inverse_Rn(Rn):
            return 1./np.clip(Rn.flatten(), a_min=1e-6, a_max=Rn.max())  

        if R_n is not None:
            if isinstance(R_n, str): #basically if it's 'model' or 'iterative':
                inv_R_n = inverse_Rn(PSF_0*flux_scale + self.readoutnoise**2)
            else:
                inv_R_n = inverse_Rn(np.array(R_n))
        else:
            inv_R_n = inverse_Rn(np.ones_like(PSF)*self.readoutnoise**2)

        for i in range(self.iterations):    
            dI = (PSF-PSF_cap).flatten()
    
            C.append( np.dot(dI*inv_R_n, dI) )  # check the convergence
            if i > 0 and (criterion(i)<1e-6 or coefs_norm(A_ests[i]-A_ests[i-1])<1e-12):
                if verbous:
                    print('Criterion', criterion(i), 'is reached at iter.', i)
                break
            
            # Generate interaction matricies
            H = self.generateLIFTinteractionMatrices(A_est, modes, flux_scale/flux_cap)

            # Maximum likelyhood estimation
            P_ML = np.linalg.pinv(H.T * inv_R_n @ H) @ H.T * inv_R_n
            d_A = P_ML @ dI
            A_est[modes] += d_A
            
            # Save the intermediate results for history
            Hs.append(H)
            P_MLs.append(P_ML)
            A_ests.append(np.copy(A_est))

            if verbous:
                print('Criterion:', criterion(i))
                self.print_modes(d_A*1e9)
                print()
            
            # Update the PSF image with the estimated coefficients
            PSF_cap, flux_cap = normalize_PSF(PSF_from_coefs(A_est))
            PSF_cap *= flux_scale
            
            if isinstance(R_n, str):
                if R_n == 'iterative':
                    inv_R_n = inverse_Rn(PSF_cap + self.readoutnoise**2)

        history = { # contains intermediate data saved at every iteration
            'P_ML' : convert( np.dstack(P_MLs) ),
            'H'    : convert( np.dstack(Hs) ),
            'A_est': convert( np.squeeze(np.dstack(A_ests), axis=0) ),
            'C'    : convert( np.array(C) )
        }
        return convert(A_est), convert(PSF_cap), history

