import numpy as np
import matplotlib.pyplot as plt
from .micro_doppler import Micro_Doppler
from .utils import calculate_power_range

class Radar():
    def __init__(self,
                T = 10, # time duration
                n_samples = 8192, # number of samples
                lambda_ = 0.03,  # Wavelength in meters
                rangeres = 0.05,  # Range resolution in meters
                radarloc = np.array([20, 0, -10]),  # Radar location
        ):

        c = 2.99792458e8  # Speed of light in m/s
        self.f0 = c / lambda_  # Frequency in Hz

        # Total number of range bins
        self.nr = int(2 * np.sqrt(np.sum(radarloc**2)) / rangeres)

        self.T = T
        self.nt = n_samples
        self.lambda_ = lambda_
        self.rangeres = rangeres
        self.radarloc = radarloc

        self.dt = T / n_samples  # Time interval

        # in-phase and quadrature components of the return signal.
        self.data = np.zeros((self.nr,n_samples), dtype=np.complex64)
        self.TF = None # time-freq -> microdoppler


    def radar_return(self,part1,part2,rcs_func,k):
        # for [part1, part2] is the vector of the body part
        center = (part1 + part2) / 2

        # for k in range(nt):
        r_dist = np.abs(center - self.radarloc)
        distance = np.sqrt(np.sum(r_dist**2))
        A = self.radarloc - center # radar direction
        B = part2 - part1          # part direction/ aspect

        ThetaAngle = np.arctan2(np.sqrt((self.radarloc[0] - part2[0])**2 + (self.radarloc[1] - part2[1])**2), self.radarloc[2] - part2[2])
        PhiAngle = -np.arctan2(self.radarloc[1] - part2[1], self.radarloc[0] - part2[0])

        rcs = rcs_func(PhiAngle, ThetaAngle, self.lambda_)
        amp = np.sqrt(rcs)
        PHs = amp * np.exp(-1j * 4 * np.pi * distance / self.lambda_)
        self.data[int(np.floor(distance / self.rangeres)),k] += PHs
    
    def simulate(self,obj, verbose = False):
        print(f"Radar simulation in progress")
        for t in range(self.nt):
            parts = obj._get_all_parts(return_rcs = True)
            for (part1,part2), rcs_func in parts: 
                self.radar_return(part1,part2,rcs_func,t)

            if verbose and t % 1000 == 0:
                    print(f"Pulse [{t}/{self.nt}]")

            # update object
            obj._update(self.dt)

    def _plot_range_profile(self, ylim = None, power_limits = None):
        # Figure 2: Range Profiles of Bird Flapping Wings
        fig, ax = plt.subplots()
        cax = ax.imshow(20 * np.log10(np.abs(self.data) + np.finfo(float).eps ),\
                         aspect='auto', cmap='jet', extent=[1, self.nt, 0, self.nr * self.rangeres])
        ax.set_xlabel('Pulses')
        ax.set_ylabel('Range (m)')
        ax.set_title('Range Profiles')
        plt.colorbar(cax)

        if ylim is not None:
            ax.set_ylim(ylim[0],ylim[1])

        if power_limits is not None:
            cax.set_clim([power_limits[0], power_limits[1]])  # Adjust the color limits

        plt.draw()
        plt.pause(2.01)
           

    def _calculate_Micro_Doppler(self,window_size = 512):
        # Summing Along Range Bins
        x = np.sum(self.data, axis=0)

        T = self.T
        F = self.nt/T

        self.TF = Micro_Doppler(x,window_size,self.nt)
        self.MD = 20 * np.log10(np.fft.fftshift(np.abs(self.TF), axes=0) + np.finfo(float).eps)


    def _plot_micro_doppler_signature(self, 
                                      window_size = 512, 
                                      power_limits = None,
                                      Hz_limits = None
                                      ):

        if self.TF is None:
            self._calculate_Micro_Doppler(window_size)

        if power_limits is None:
            power_limits = calculate_power_range(self.MD)
            print(f'automated power limits = {power_limits}')
            # power_limits[1]=0
            # print(f'setting upper limit to zero')

        T = self.T
        F = self.nt/T

        # Display final time-frequency signature
        fig, ax = plt.subplots()
        cax = ax.imshow(self.MD\
                        , aspect='auto', cmap='jet', extent=[0, T, -F / 2, F / 2])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Doppler (Hz)')

        if power_limits is not None:
            clim = cax.get_clim()
            cax.set_clim([power_limits[0],power_limits[1]])  # Adjust the color limits
            # cax.set_clim([clim[1] +power_limits[0], clim[1] +power_limits[1]])  # Adjust the color limits

        if Hz_limits:
            ax.set_ylim(Hz_limits)

        plt.title('Micro-Doppler Signature')
        plt.colorbar(cax)
        plt.show()
        plt.pause(2.01)



