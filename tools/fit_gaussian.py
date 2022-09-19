# %%
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import pi, r_
from scipy import optimize

'''radius = 0.002
OPD = np.pi / 8.0

x      = np.linspace(-radius, radius, 100)
xx, yy = np.meshgrid(x, x)
#circle = xx**2 + yy**2

def astigmatismPSF(xx, yy, r):
    r_stretched = 2*r/np.sqrt(2.0) 
    R =  xx * 0 + r_stretched
    z = np.sqrt(R**2 - xx * yy - (xx**2 + yy**2)*0.5)
    z[np.isnan(z)] = 0
    return z / r_stretched


z = cylinder(xx, yy, radius) * OPD
plt.imshow(z, extent = [np.min(xx), np.max(xx), np.min(yy), np.max(yy)])'''


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp( -(((center_x-x)/width_x)**2 + ((center_y-y)/width_y)**2)/2 )


def gaussian_fourier(height, center_u, center_v, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda u,v: height*np.exp( -(((center_u-u)*width_x)**2 + ((center_v-v)*width_y)**2) * 2 * np.pi**2 )


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def plot_gaussian(data, params, extents=None):
    fit = gaussian(*params)

    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper, extent=extents)
    ax = plt.gca()
    (height, x, y, width_x, width_y) = params

    FWHM_x = 2*np.sqrt(2*np.log(2)) * width_x
    FWHM_y = 2*np.sqrt(2*np.log(2)) * width_y

    plt.text(0.95, 0.05, """
    x : %.1f
    y : %.1f
    FWHM$_x$ : %.1f
    FWHM$_y$ : %.1f""" %(x, y, FWHM_x, FWHM_y), fontsize=16, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)


def FitAndPlotGauss1D(x, y, label=None):
    n = len(x)                          #the number of data
    A = y.max()
    w = y / y.max()

    mean  = np.sum(x*w) / np.sum(w)
    sigma = np.sqrt( sum(w*(x-mean)**2) / ( (n-1)/n * np.sum(w) ) )
        
    def gaus(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    popt, pcov = optimize.curve_fit(gaus, x, y, p0=[A,mean,sigma])
    #plt.plot(x, gaus(x, *popt), 'r:', label=label)    
    #print("Sigma:", popt[2])
    return popt

# %%
# Create the gaussian data
def fit_test():
    Xin, Yin = np.mgrid[0:201, 0:201]
    data = gaussian(3, 100, 100, 20, 40)(Xin, Yin) + np.random.random(Xin.shape)

    plt.matshow(data, cmap=plt.cm.gist_earth_r)

    params = fitgaussian(data)
    fit = gaussian(*params)

    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    ax = plt.gca()
    (height, x, y, width_x, width_y) = params

    plt.text(0.95, 0.05, """
    x : %.1f
    y : %.1f
    width_x : %.1f
    width_y : %.1f""" %(x, y, width_x, width_y),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)
