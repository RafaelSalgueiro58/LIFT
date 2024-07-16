import numpy as np
from numpy.random import Generator, PCG64
from scipy import signal as sg

try:
    import cupy as cp
except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp = np

# Detector based on Arseniy that is going to be used as a complement to OOPAO

class Detector_Rafael:
    def __init__(self, tel, pixel_size, ang_pixel, img_resolution, samples=1, RON=0, QE=1):
        self.QE             = QE                                                 #TODO: use QE?
        self.pixel_size     = pixel_size                                         # m
        self.readoutNoise   = RON                                                # STD of the readout noise
        self.samples        = samples   
        self.ang_pixel      = ang_pixel                                          # rad
        self.f              = self.pixel_size / self.ang_pixel * 206264806.71915 # [m]
        self.img_resolution = img_resolution
        self.object         = None


    def getFrame(self, PSF_inp, noise=True, integrate=True):

        PSF = cp.asnumpy(PSF_inp) if hasattr(PSF_inp, 'device') else np.copy(PSF_inp)

        if self.object is not None:
            PSF = sg.convolve2d(PSF, self.object, boundary='symm', mode='same') / self.object.sum()

        if noise: 
            R_n = PSF + self.readoutNoise
            rng = np.random.default_rng()
            rg  = Generator(PCG64())
        else:
            R_n = np.ones(PSF.shape)
        
        # Compose the image cube
        image_cube = []
        for i in range(self.samples):
            if noise:
                photon = rng.poisson(PSF, PSF.shape)
                ron = rg.standard_normal(PSF.shape) * np.sqrt(self.readoutNoise)
                image_cube.append(photon + ron)
            else:
                image_cube.append(PSF)

        image_cube = np.dstack(image_cube)

        if integrate:
            image_cube = image_cube.mean(axis=2)
            if noise:
                R_n /= self.samples 

        return image_cube, R_n

