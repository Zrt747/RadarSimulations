import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import display, clear_output
from .utils import is_jupyter

class Drone():
    """
    A class to represent a drone with rotating blades.

    Attributes
    ----------
    N_blades : int
        The number of blades (2 or 3). Default is 2.
    blade_length : float
        The length of each blade in meters. Default is 6.
    blade_width : float
        The width of each blade in meters. Default is 1.
    blade_root : float
        The distance from the center to the start of the blade in meters. Default is 0.5.
    omega : float
        The rotation rate of the blades in radians per second. Default is 4 * 2 * np.pi.
    initial_position : np.ndarray
        The initial position of the drone in 3D space as a NumPy array [x, y, z]. Default is [0, 0, 0].
    velocity : np.ndarray - not implemented.....YET
        The velocity of the drone in 3D space as a NumPy array [vx, vy, vz]. Default is None.

    """
    def __init__(self,
                 N_blades = 2,  # number of blades (2 or 3)
                 blade_length = 6,  # blade length (m)
                 blade_width = 1,  # blade width (m)
                 blade_root = 0.5,  # distance from the center to start the blade (m)
                 omega = 4 * 2 * np.pi, # blade rotation rate
                 initial_position = np.array([0, 0, 0]), # need to implement
                 velocity = None,                       # need to implement
                 ):
        

        # Rotor blades parameters
        self.rotorloc = initial_position  # rotor center location
        self.phi = 0 # initial angel
        self.Nb = N_blades  # number of blades (2 or 3)
        self.L = blade_length  # blade length (m)
        self.W = blade_width  # blade width (m)
        self.L1 = blade_root  # one end (root) of blade (m)
        self.L2 = blade_root + blade_length  # other end (tip) of blade (m)
        self.Omega = omega  # blade rotation rate
        self.a = (self.L2 - blade_root) / 2  # rectangular blade parameter
        self.b = blade_width / 2  # rectangular blade parameter
        self.tip_loc = self.rotorloc + np.array([blade_root+blade_length, 0, 0]) # location of the tip of the blade
        self.root_loc = self.rotorloc + np.array([blade_root, 0, 0]) # location of the tip of the blade

    def _update(self,dt):
        # update the currect angle for given time step
        self.phi += dt*self.Omega 

    def _get_blade(self,blade_id, return_rcs = False):
        Rzxz = x_convention(self.phi + blade_id*(2 * np.pi / self.Nb), 0, 0) 
        part1 = Rzxz @ self.root_loc
        part2 = Rzxz @ self.tip_loc
        position = np.array([part1, part2])
        # position = np.array([self.rotorloc, part2])
        if return_rcs:
            # return position, lambda phi, theta,lambda_: rcs_rect(self.a, self.b, phi, theta,lambda_)
            return position, self._blade_rcs_func()
        else:
            return position

    def _blade_rcs_func(self):
        return lambda phi, theta,lambda_: rcs_rect(self.a, self.b, phi, theta,lambda_)
        # return rcs_rect(self.a, self.b, phi, theta,lambda_)

    def _get_all_parts(self,return_rcs = False):
        parts = []
        for blade_id in range(self.Nb):
            parts.append(self._get_blade(blade_id,return_rcs))
        return parts

    def _plot_rect_blade(self,blades_points,ax):
        # fix blade root!
        for point1, point2 in blades_points:
            # Calculate the direction vector of the blade
            direction_vector = np.array(point2) - np.array(point1)
            direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize

            # Find a perpendicular vector in the xy-plane
            perpendicular_vector = np.array([-direction_vector[1], direction_vector[0], 0])
            perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)  # Normalize

            # Define the corners of the rectangle (blade)
            half_width = self.W / 2
            corner1 = point1 + half_width * perpendicular_vector
            corner2 = point1 - half_width * perpendicular_vector
            corner3 = point2 - half_width * perpendicular_vector
            corner4 = point2 + half_width * perpendicular_vector

            # Create the polygon for the blade
            blade_corners = [corner1, corner2, corner3, corner4]
            poly3d = Poly3DCollection([blade_corners], color='k', linewidths=2, edgecolors='k', alpha = 0.8)
            ax.add_collection3d(poly3d)

    def Plot3D(
        # 3D plot of the bird
                self
            ,T = 20  # Total observation time
            ,dt = 0.02  # Time interval
            ):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for _ in range(T): #range(0, 60, 5):

            self._update(dt)
            verts = self._get_all_parts()
            self._plot_rect_blade(verts,ax)
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Drone simualation')
            ax.grid(True)

            # plot "body" for fun - this is not a part - yet
            a = 2  # Radius along the x-axis
            b = 1  # Radius along the y-axis
            c = 1  # Radius along the z-axis
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = a * np.outer(np.cos(u), np.sin(v))
            y = b * np.outer(np.sin(u), np.sin(v))
            z = c * np.outer(np.ones_like(u), np.cos(v)) - c
            ax.plot_surface(x, y, z, color='k', alpha=0.8)

            if is_jupyter():
                clear_output(wait=True)  # Clear the current output
                display(fig)  # Display the updated figure
                plt.pause(0.01)  # Pause to create animation effect
                ax.clear()
                
            else:
                plt.draw()
                plt.pause(0.01)


def x_convention(phi, theta, psi):
    """
    x-convention: Z-X-Z sequence

    phi angle: rotation about z-axis
    theta angle: rotation about x-axis
    psi angle: rotation about z-axis

    Returns:
    Rzxz - combined rotation matrix
    """
    # Rotation matrix about the z-axis by an angle phi
    R1 = np.array([
        [np.cos(phi), np.sin(phi), 0],
        [-np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    
    # Rotation matrix about the x-axis by an angle theta
    R2 = np.array([
        [1, 0, 0],
        [0, np.cos(theta), np.sin(theta)],
        [0, -np.sin(theta), np.cos(theta)]
    ])
    
    # Rotation matrix about the z-axis by an angle psi
    R3 = np.array([
        [np.cos(psi), np.sin(psi), 0],
        [-np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix in Z-X-Z convention
    Rzxz = R3 @ (R2 @ R1)
    
    return Rzxz


def rcs_rect(a, b, phi, theta, lambda_):
    """
    This function computes the backscattered RCS for a rectangular flat plate 
    using the Physical Optics approximation (Eq. 3.37).
    
    Parameters:
    a - length of the plate along the x-axis
    b - length of the plate along the y-axis
    theta - elevation angle in radians
    phi - azimuth angle in radians
    lambda_  - wavelength of the radar signal (m)
    
    Returns:
    rcs - backscattered Radar Cross Section (RCS)
    """
    eps = 1e-6
    ka = 2 * np.pi * a / lambda_
    kb = 2 * np.pi * b / lambda_
    angle_a = ka * np.sin(theta) * np.cos(phi)
    angle_b = kb * np.sin(theta) * np.sin(phi)
    
    rcs = (4 * np.pi * a**2 * b**2 / lambda_**2) * (np.cos(theta)**2) * \
          ((np.sin(angle_a) * np.sin(angle_b) / (angle_a * angle_b))**2) + eps
    
    return rcs

