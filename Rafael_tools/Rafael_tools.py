import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
    global_gpu_flag = True
except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp = np
    global_gpu_flag = False

def Pad_array(l1,l2,array):
    
    # Determine the required padding
    rows_to_pad = l1 - l2
    cols_to_pad = l1 - l2
    
    # Pad the array with zeros
    if rows_to_pad > 0 or cols_to_pad > 0:
        row_padding = rows_to_pad // 2
        col_padding = cols_to_pad // 2
        array_padded = np.pad(array, ((row_padding, rows_to_pad - row_padding), (col_padding, cols_to_pad - col_padding)), mode='constant')
    else:
        array_padded = array  # No padding needed

    return array_padded
    

def extractor(array,l):
    
    center_x = int(array.shape[0] // 2 - l/2) 
    center_y = int(array.shape[1] // 2 - l/2)
    array_extracted = array[center_x:center_x+l, center_y:center_y+l]

    return array_extracted

def make_3_plot_graph(data_1,title_1,data_2,title_2,data_3,title_3):

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # Adjust the figure size as needed
    plt.subplots_adjust(wspace=0.5)  # Adjust the horizontal spacing between subplots
    
    # Plot on each subplot
    img0 = axs[0].imshow(data_1)
    axs[0].set_title(title_1)
    fig.colorbar(img0, ax=axs[0])
    
    img1 = axs[1].imshow(data_2)
    axs[1].set_title(title_2)
    fig.colorbar(img1, ax=axs[1])
    
    img2 = axs[2].imshow(data_3)
    axs[2].set_title(title_3)
    fig.colorbar(img2, ax=axs[2])
    plt.show()

# To make this function work, one must ensure that size of inp can be divided by N
def binning(inp, N):
    if N == 1: return inp
    xp = cp if hasattr(inp, 'device') else np
    out = xp.dstack(xp.split(xp.dstack(xp.split(inp, inp.shape[0]//N, axis=0)), inp.shape[1]//N, axis=1))
    return out.sum(axis=(0,1)).reshape([inp.shape[0]//N, inp.shape[1]//N]).T


def PropagateField(tel, det, amplitude, phase, wavelength, return_intensity, oversampling=1):
    
    xp = np

    zeroPaddingFactor = det.f / det.pixel_size * wavelength / tel.D
    resolution = tel.pupil.shape[0]
    
    if oversampling is not None: oversampling = oversampling

    if det.img_resolution > zeroPaddingFactor*resolution:
        print('Error: image has too many pixels for this pupil sampling. Try using a pupil mask with more pixels')
        return None

    # If PSF is undersampled apply the integer oversampling
    if zeroPaddingFactor * oversampling < 2:
        oversampling = (np.ceil(2.0/zeroPaddingFactor)).astype('int')
    
    # This is to ensure that PSF will be binned properly if number of pixels is odd
    if oversampling % 2 != det.img_resolution % 2:
        oversampling += 1

    img_size = np.ceil(det.img_resolution*oversampling).astype('int')
    N = np.fix(zeroPaddingFactor * oversampling * resolution).astype('int')
    pad_width = np.ceil((N-resolution)/2).astype('int')

    if not hasattr(amplitude, 'device'): amplitude = xp.array(amplitude, dtype=cp.float32)
    if not hasattr(phase, 'device'):     phase     = xp.array(phase, dtype=cp.complex64)
    
    #supportPadded = cp.pad(amplitude * cp.exp(1j*phase), pad_width=pad_width, constant_values=0)
    supportPadded = xp.pad(amplitude * xp.exp(1j*phase), pad_width=((pad_width,pad_width),(pad_width,pad_width)), constant_values=0)
    N = supportPadded.shape[0] # make sure the number of pxels is correct after the padding

    # PSF computation
    [xx,yy] = xp.meshgrid( xp.linspace(0,N-1,N), xp.linspace(0,N-1,N), copy=False )    
    center_aligner = xp.exp(-1j*xp.pi/N * (xx+yy) * (1-det.img_resolution%2)).astype(xp.complex64)
    #                                                        ^--- this is to account odd/even number of pixels
    # Propagate with Fourier shifting
    EMF = xp.fft.fftshift(1/N * xp.fft.fft2(xp.fft.ifftshift(supportPadded*center_aligner)))

    # Again, this is to properly crop a PSF with the odd/even number of pixels
    if N % 2 == img_size % 2:
        shift_pix = 0
    else:
        if N % 2 == 0: shift_pix = 1
        else: shift_pix = -1

    # Support only rectangular PSFs
    ids = xp.array([np.ceil(N/2) - img_size//2+(1-N%2)-1, np.ceil(N/2) + img_size//2+shift_pix]).astype(xp.int32)
    EMF = EMF[ids[0]:ids[1], ids[0]:ids[1]]

    if return_intensity:
        return binning(xp.abs(EMF)**2, oversampling)

    return EMF,oversampling # in this case, raw electromagnetic field is returned. It can't be simply binned



def Propagate_field(zeroPaddingFactor,amplitude,phase,tel):

    N = int(zeroPaddingFactor*tel.resolution)

    center  = N//2           
    norma   = N
    mask = 1
    amp_mask = 1

    # zeroPadded support for the FFT
    supportPadded = np.zeros([N,N],dtype='complex')
    supportPadded [center-tel.resolution//2:center+tel.resolution//2,center-tel.resolution//2:center+tel.resolution//2] \
    = amp_mask*tel.pupil*tel.pupilReflectivity*amplitude*np.exp(1j*phase)
    [xx,yy]                         = np.meshgrid(np.linspace(0,N-1,N),np.linspace(0,N-1,N))
    phasor                          = np.exp(-(1j*np.pi*(N+1)/N)*(xx+yy))

    EMF = np.fft.fft2(np.fft.fftshift(supportPadded*phasor))*mask/norma
    #EMF = np.fft.fftshift(supportPadded*tel.phasor)*mask/norma

    return EMF # electromagnetic field