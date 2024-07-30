import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import display, clear_output
from .utils import is_jupyter

class Bird():
    """
    A class to represent a bird with flapping wing mechanics.

    Attributes
    ----------
    initial_position : list of float
        Initial position of the bird in 3D space [x, y, z] (default is [0, 0, 0]).
    velocity : list of float
        Linear velocity vector of the bird [vx, vy, vz] (default is [0.2, 0, 0]).
    body_length : float
        Length of the bird's body (default is 0.8).
    L1 : float
        Length of wing segment 1 (default is 0.5).
    L2 : float
        Length of wing segment 2 (default is 0.5).
    A1 : float
        Amplitude of flapping angle for segment 1 in degrees (default is 40).
    A2 : float
        Amplitude of flapping angle for segment 2 in degrees (default is 30).
    ksi10 : float
        Lag flapping angle for segment 1 in degrees (default is 15).
    ksi20 : float
        Lag flapping angle for segment 2 in degrees (default is 40).
    C2 : float
        Amplitude of twisting angle for segment 2 in degrees (default is 20).
    F0 : float
        Flapping frequency (default is 1).

    """
    def __init__(self
                 # Initial conditions
                ,initial_position = [0,0,0] # initial position 
                ,velocity = [0.2,0,0]  # Linear velocity vector

                # Bird Dimensions
                ,body_length = 0.8
                ,wing_len_1 = 0.5  # Length of segment 1
                ,wing_len_2 = 0.5  # Length of segment 2

                # flapping speed and amplitude
                ,A1 = 40  # Amplitude of flapping angle in degrees
                ,A2 = 30  # Amplitude of segment2 flapping angle
                ,ksi10 = 15  # Lag flapping angle in degrees
                ,ksi20 = 40  # Lag flapping angle in degrees
                ,C2 = 20  # Amplitude of segment2 twisting angle
                ,F0 = 1  # Flapping frequency

                 ) -> None:
        
        # save local variables
        self.F0 = F0
        self.A1 = A1
        self.speed = np.linalg.norm(velocity)
        self.ksi10 = ksi10
        self.L1 = wing_len_1
        self.A2 = A2
        self.ksi20 = ksi20
        self.L2 = wing_len_2
        self.C2 = C2

        # create trasformation for non simplified xyz system in the direction of the velocity vector
        self.T = transformation_matrix(normalize(velocity))
        self.initial_position = initial_position
        self._body_vector = self.T @ np.array([body_length/2,0.,0.])

        # initial bird key attributes that change in time
        self._body_center = np.array([0,0,0], dtype=np.float32)
        self.flap_angel = 0
        self.ksi1 = A1 + ksi10 # Initial flapping angle
        self.ksi2 = A2 + ksi20
        self.theta2 = 0

        # initializing parts
        x1,y1,z1,x2,y2,z2 = self._get_xyz()
        self._update_body_parts(x1,y1,z1,x2,y2,z2)
        
    
    def _update(self,dt):
        """
        updates bird location for given time step

        Attributes
        ----------
        dt : float
        Time step increment
        """

        # Update body center and angles
        self._body_center += self.speed * np.array([1, 0, 0])* dt
        self.flap_angel += 2 * np.pi * self.F0 * dt
        self.ksi1 = self.A1 * np.cos(self.flap_angel) + self.ksi10 # flapping angle
        self.ksi2 = self.A2 * np.cos(self.flap_angel) + self.ksi20
        self.theta2 = self.C2 * np.sin(self.flap_angel)

        # get key coordinates
        x1,y1,z1,x2,y2,z2 = self._get_xyz()
        self._update_body_parts(x1,y1,z1,x2,y2,z2)

    def _get_xyz(self):
        """
        sub function to calculate the coordinates of the wings critical points

        """
        x1= self._body_center[0]
        y1 = self.L1 * np.cos(np.radians(self.ksi1))
        z1 = y1 * np.tan(np.radians(self.ksi1)) 

        d = self.theta2 / np.cos(np.radians(self.ksi1 - self.ksi2))
        y2 = self.L1 * np.cos(np.radians(self.ksi1)) +\
              self.L2 * np.cos(np.radians(self.theta2)) * np.cos(np.radians(self.ksi1 - self.ksi2))
        x2 = x1 - (y2 - y1) * np.tan(np.radians(d))
        z2 = z1 + (y2 - y1) * np.tan(np.radians(self.ksi1 - self.ksi2))
        return x1,y1,z1,x2,y2,z2 


    def _update_body_parts(self,x1,y1,z1,x2,y2,z2):
        """
        Update body parts location in transform corrdinates
        """
        self._bc = self._coo_transform(self._body_center)
        self._lw1 = self._coo_transform(np.array([x1, y1, z1]))
        self._lw2 = self._coo_transform(np.array([x2, y2, z2]))
        self._rw1 = self._coo_transform(np.array([x1, -y1, z1]))
        self._rw2 = self._coo_transform(np.array([x2, -y2, z2]))

    def _coo_transform(self,x):
        """
        Transfrom from centrelize coordinates system to user selected
        """
        return (self.T @ x) + self.initial_position
    
    def _body(self,return_rcs = False):
        position = np.array([self._bc+self._body_vector, self._bc-self._body_vector])
        return [position, self._wing_rcs_func] if return_rcs else position

    def _left_wing_1(self,return_rcs = False):
        position = np.stack([self._bc,self._lw1])
        return [position, self._wing_rcs_func] if return_rcs else position
    
    def _left_wing_2(self,return_rcs = False):
        position = np.stack([self._lw1,self._lw2])
        return [position, self._wing_rcs_func] if return_rcs else position
    
    def _right_wing_1(self,return_rcs = False):
        position = np.stack([self._bc,self._rw1])
        return [position, self._wing_rcs_func] if return_rcs else position

    def _right_wing_2(self,return_rcs = False):
        position = np.stack([self._rw1,self._rw2])
        return [position, self._wing_rcs_func] if return_rcs else position

    
    def _get_all_parts(self,return_rcs = False):
        return [self._body(return_rcs) ,self._left_wing_1(return_rcs), self._left_wing_2(return_rcs), \
                self._right_wing_1(return_rcs), self._right_wing_2(return_rcs)]
    
    @staticmethod
    def _wing_rcs_func(phi, theta,lambda_):
        return rcsellipsoid(0.05, 0.05, 0.25, phi, theta,lambda_ )
    
    @staticmethod
    def _body_rcs_func(phi, theta,lambda_):
        return rcsellipsoid(0.1, 0.1, 1.0, phi, theta,lambda_ )

    def Plot3D(
        # 3D plot of the bird
                self
               ,T = 100  # Total observation time
               ,dt = 0.05  # Time interval
               ,plot_trail = False
               ):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        trailx = []  
        traily = []
        trailz = []

        for _ in range(T): #range(0, 60, 5):

            self._update(dt)
            verts = self._get_all_parts()

            if plot_trail:
                tx, ty, tz = self._bc
                trailx.append(tx)
                traily.append(ty)
                trailz.append(tz)
                ax.plot(trailx, traily, trailz,'ro')

            for vert in verts:
                poly3d = Poly3DCollection([vert], color='k', linewidths=2)
                ax.add_collection3d(poly3d)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Bird Wing Flapping')
            ax.grid(True)

            if is_jupyter():
                clear_output(wait=True)  # Clear the current output
                display(fig)  # Display the updated figure
                plt.pause(0.01)  # Pause to create animation effect
                ax.clear()
                
            else:
                plt.draw()
                plt.pause(0.01)


def rcsellipsoid(a, b, c, phi, theta,lambda_ ):
    return (a * b * c)**2 / (a**2 * np.sin(phi)**2 * np.cos(theta)**2 + b**2 * np.sin(phi)**2 * np.sin(theta)**2 + c**2 * np.cos(phi)**2)


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm

# Function to construct the transformation matrix
def transformation_matrix(vector):
    # Normalize the given vector to be the new x-axis
    u_x = normalize(vector)
    
    # Choose an arbitrary vector that is not parallel to u_x
    if np.allclose(u_x, [0, 0, 1]):
        arbitrary_vector = np.array([0, 1, 0])
    else:
        arbitrary_vector = np.array([0, 0, 1])
    
    # Calculate the new y-axis using the cross product and normalize it
    u_y = np.cross(u_x, arbitrary_vector)
    u_y = normalize(u_y)
    
    # Calculate the new z-axis using the cross product of u_x and u_y
    u_z = -np.cross(u_x, u_y)
    
    # Construct the transformation matrix
    # transformation_mat = np.array([u_x, u_y, u_z])
    transformation_mat = np.column_stack((u_x, u_y, u_z))
    
    return transformation_mat.T  # Transpose to get the correct orientation