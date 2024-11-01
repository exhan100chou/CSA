import torch 
import numpy as np
import math
class APSK:
    def __init__(self, M, PSNR):
        self.M = M
        self.r1 = 1  # Radius of the inner ring
        self.r2 = 2  # Radius of the outer ring
        self.r3 = 3  # Radius of the outer ring
        self.constellation = self.build_constellation()
        self.p = self.calculate_p() 
        self.delta = self.compute_noise(PSNR)
   
    def calculate_p(self):
        if self.M == 16 :
           # Mean power for 16APSK (considering the radii)
           p_inner = self.r1**2 * 4
           p_outer = self.r2**2 * 12
           return (p_inner + p_outer) / 16
        elif self.M == 32 :
           # Mean power for 32APSK (considering the radii)
           p_inner = self.r1**2 * 4
           p_middle = self.r2**2 * 12
           p_outer = self.r3**2 * 16
           return (p_inner + p_middle + p_outer) / 32
   
    def compute_noise(self, PSNR):
        # Compute noise standard deviation
        delta_2 = self.p / torch.pow(torch.tensor(10), PSNR / 10).float()
        return torch.sqrt(delta_2 / 2)
   
    def build_constellation(self):
        if self.M == 16 :        
           # Inner ring (4 points)
           inner_points = [(self.r1 * np.cos(2 * np.pi * i / 4), self.r1 * np.sin(2 * np.pi * i / 4)) for i in range(4)]
       
           # Outer ring (12 points)
           outer_points = [(self.r2 * np.cos(2 * np.pi * i / 12), self.r2 * np.sin(2 * np.pi * i / 12)) for i in range(12)]
       
           # Combine the points
           constellation = inner_points + outer_points
           return torch.tensor(constellation)
        elif self.M == 32 :
            # Inner ring (4 points)
            inner_points = [(self.r1 * np.cos(2 * np.pi * i / 4), self.r1 * np.sin(2 * np.pi * i / 4)) for i in range(4)]
        
            # Middle ring (12 points)
            middle_points = [(self.r2 * np.cos(2 * np.pi * i / 12), self.r2 * np.sin(2 * np.pi * i / 12)) for i in range(12)]
       
            # Outer ring (16 points)
            outer_points = [(self.r3 * np.cos(2 * np.pi * i / 16), self.r3 * np.sin(2 * np.pi * i / 16)) for i in range(16)]
       
            # Combine the points
            constellation = inner_points + middle_points + outer_points
            return torch.tensor(constellation)   
    def modulate(self, z: torch.Tensor):
        # Convert input symbols to 16APSK constellation points
        m = z.shape[0]
        X = torch.zeros(m, 2)
        for i in range(m):
            X[i] = self.constellation[int(z[i])]
        return X
   
    def awgn(self, X):
        # Add AWGN noise to the signal
        X += self.delta * torch.randn_like(X)
        return X
   
    def demodulate(self, X):
        #X = X.to(torch.float32)  # Ensure X is float32
       
        # Convert tensors to numpy arrays for compatibility with np functions
        X_np = X.numpy()
        constellation_np = self.constellation.numpy()
       
        # Compute squared distances
        distances = np.linalg.norm(X_np[:, np.newaxis] - constellation_np, axis=2)
       
        # Find the index of the minimum distance for each received signal
        return torch.tensor(np.argmin(distances, axis=1), dtype=torch.long)
    def rayleigh_fading(self, X):
        # Rayleigh fading channel coefficients
        h = torch.sqrt(torch.tensor(1/2.0)) * torch.randn_like(X) + 1j * torch.randn_like(X)
        
        # Apply the Rayleigh fading channel
        X_faded = h * X
        
        # Add AWGN noise
        X_faded = self.awgn(X_faded)
        
        return X_faded   
    def rician_fading(self, X):
        K =2.8
        # Line-of-Sight (LOS) component
        h_los = torch.sqrt(torch.tensor(K / (K + 1)))
        
        # Scattered component (non-LOS)
        h_scatter = torch.sqrt(torch.tensor(1 / (K + 1) / 2.0)) * (torch.randn_like(X) + 1j * torch.randn_like(X))
        
        # Rician fading channel coefficient
        h = h_los + h_scatter
        
        # Apply the Rician fading channel
        X_faded = h * X
        
        # Add AWGN noise
        X_faded = self.awgn(X_faded)
        
        return X_faded        
    def leo_channel_rician(self, X): 
        K =2.8
        user_lat = 40.7128  # User latitude (in degrees)
        user_lon = -74.0060  # User longitude (in degrees)
        user_alt = 0.0  # User altitude (in meters, assume sea level)

        sat_lat = 0.0  # Satellite latitude (in degrees)
        sat_lon = 0.0  # Satellite longitude (in degrees)
        sat_alt = 600e3  # Satellite altitude (in meters, LEO orbit)

        tx_gain = 35  # Transmitter antenna gain in dBi
        rx_gain = 37  # Receiver antenna gain in dBi

        frequency = 28e9  # Carrier frequency in Hz (28 GHz)
        R = 6371e3  # Earth's radius in meters
        (lat1, lon1, lat2, lon2, alt1, alt2)=(user_lat, user_lon, sat_lat, sat_lon, user_alt, sat_alt)
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        ground_distance = R * c  # Distance over Earth's surface
        distance = np.sqrt(ground_distance**2 + (alt2 - alt1)**2)  # 3D distance considering altitude
        c = 3e8  # Speed of light in m/s
        fspl = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)   
           # Convert gains from dB to linear scale
        tx_gain_linear = 10**(tx_gain / 10)
        rx_gain_linear = 10**(rx_gain / 10)
   
    # Convert FSPL from dB to linear scale
        fspl_linear = 10**(fspl / 10)
        
    # Calculate the channel coefficient (linear scale)
        channel_coeff = torch.sqrt(torch.tensor(tx_gain_linear * rx_gain_linear / fspl_linear))

        h_los = torch.sqrt(torch.tensor(K / (K + 1)))
        
        # Scattered component (non-LOS)
        h_scatter = torch.sqrt(torch.tensor(1 / (K + 1) / 2.0)) * (torch.randn_like(X) + 1j * torch.randn_like(X))
        
        # Rician fading channel coefficient
        h = h_los + h_scatter
     #   X = self.awgn(X)
        X_leo = channel_coeff * h * X 
        X_leo = self.awgn(X_leo)
     # Assume MMSE signal recovery   
        X_leo = X_leo / channel_coeff 
        return X_leo
    def leo_channel_rayleigh(self, X): 
        user_lat = 40.7128  # User latitude (in degrees)
        user_lon = -74.0060  # User longitude (in degrees)
        user_alt = 0.0  # User altitude (in meters, assume sea level)

        sat_lat = 0.0  # Satellite latitude (in degrees)
        sat_lon = 0.0  # Satellite longitude (in degrees)
        sat_alt = 600e3  # Satellite altitude (in meters, LEO orbit)

        tx_gain = 35  # Transmitter antenna gain in dBi
        rx_gain = 37  # Receiver antenna gain in dBi

        frequency = 28e9  # Carrier frequency in Hz (28 GHz)
        R = 6371e3  # Earth's radius in meters
        (lat1, lon1, lat2, lon2, alt1, alt2)=(user_lat, user_lon, sat_lat, sat_lon, user_alt, sat_alt)
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        ground_distance = R * c  # Distance over Earth's surface
        distance = np.sqrt(ground_distance**2 + (alt2 - alt1)**2)  # 3D distance considering altitude
        c = 3e8  # Speed of light in m/s
        fspl = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)   
           # Convert gains from dB to linear scale
        tx_gain_linear = 10**(tx_gain / 10)
        rx_gain_linear = 10**(rx_gain / 10)
   
    # Convert FSPL from dB to linear scale
        fspl_linear = 10**(fspl / 10)
        
    # Calculate the channel coefficient (linear scale)
        channel_coeff = torch.sqrt(torch.tensor(tx_gain_linear * rx_gain_linear / fspl_linear))
        # Rayleigh fading channel coefficients
        h = torch.sqrt(torch.tensor(1/2.0)) * torch.randn_like(X) + 1j * torch.randn_like(X)
        
        # Apply the Rayleigh fading channel

        X_leo = channel_coeff * h * X 
        X_leo = self.awgn(X_leo)
     # Assume MMSE signal recovery   
        X_leo = X_leo / channel_coeff 
        return X_leo                
class PSK:
    def __init__(self, M, PSNR):
        self.M = M
        self.p = 1
        self.delta = self.compute_noise(PSNR)
        self.constellation = self.build()
    
    def compute_noise(self, PSNR):      
        delta_2 = self.p/torch.pow(torch.tensor(10), PSNR/10).float()  
        return torch.sqrt(delta_2/2)   
    def awgn(self, X):
        X += self.delta*torch.randn_like(X)
        return X
    def rayleigh_fading(self, X):
        # Rayleigh fading channel coefficients
        h = torch.sqrt(torch.tensor(1/2.0)) * torch.randn_like(X) + 1j * torch.randn_like(X)
        
        # Apply the Rayleigh fading channel
        X_faded = h * X
        
        # Add AWGN noise
        X_faded = self.awgn(X_faded)
        
        return X_faded   
    def rician_fading(self, X):
        K =2.8
        # Line-of-Sight (LOS) component
        h_los = torch.sqrt(torch.tensor(K / (K + 1)))
        
        # Scattered component (non-LOS)
        h_scatter = torch.sqrt(torch.tensor(1 / (K + 1) / 2.0)) * (torch.randn_like(X) + 1j * torch.randn_like(X))
        
        # Rician fading channel coefficient
        h = h_los + h_scatter
        
        # Apply the Rician fading channel
        X_faded = h * X
        
        # Add AWGN noise
        X_faded = self.awgn(X_faded)
        
        return X_faded        
    def leo_channel_rician(self, X): 
        K =2.8
        user_lat = 40.7128  # User latitude (in degrees)
        user_lon = -74.0060  # User longitude (in degrees)
        user_alt = 0.0  # User altitude (in meters, assume sea level)

        sat_lat = 0.0  # Satellite latitude (in degrees)
        sat_lon = 0.0  # Satellite longitude (in degrees)
        sat_alt = 600e3  # Satellite altitude (in meters, LEO orbit)

        tx_gain = 35  # Transmitter antenna gain in dBi
        rx_gain = 37  # Receiver antenna gain in dBi

        frequency = 28e9  # Carrier frequency in Hz (28 GHz)
        R = 6371e3  # Earth's radius in meters
        (lat1, lon1, lat2, lon2, alt1, alt2)=(user_lat, user_lon, sat_lat, sat_lon, user_alt, sat_alt)
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        ground_distance = R * c  # Distance over Earth's surface
        distance = np.sqrt(ground_distance**2 + (alt2 - alt1)**2)  # 3D distance considering altitude
        c = 3e8  # Speed of light in m/s
        fspl = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)   
           # Convert gains from dB to linear scale
        tx_gain_linear = 10**(tx_gain / 10)
        rx_gain_linear = 10**(rx_gain / 10)
   
    # Convert FSPL from dB to linear scale
        fspl_linear = 10**(fspl / 10)
        
    # Calculate the channel coefficient (linear scale)
        channel_coeff = torch.sqrt(torch.tensor(tx_gain_linear * rx_gain_linear / fspl_linear))

        h_los = torch.sqrt(torch.tensor(K / (K + 1)))
        
        # Scattered component (non-LOS)
        h_scatter = torch.sqrt(torch.tensor(1 / (K + 1) / 2.0)) * (torch.randn_like(X) + 1j * torch.randn_like(X))
        
        # Rician fading channel coefficient
        h = h_los + h_scatter
     #   X = self.awgn(X)
        X_leo = channel_coeff * h * X 
        X_leo = self.awgn(X_leo)
     # Assume MMSE signal recovery   
        X_leo = X_leo / channel_coeff 
        return X_leo
    def leo_channel_rayleigh(self, X): 
        user_lat = 40.7128  # User latitude (in degrees)
        user_lon = -74.0060  # User longitude (in degrees)
        user_alt = 0.0  # User altitude (in meters, assume sea level)

        sat_lat = 0.0  # Satellite latitude (in degrees)
        sat_lon = 0.0  # Satellite longitude (in degrees)
        sat_alt = 600e3  # Satellite altitude (in meters, LEO orbit)

        tx_gain = 35  # Transmitter antenna gain in dBi
        rx_gain = 37  # Receiver antenna gain in dBi

        frequency = 28e9  # Carrier frequency in Hz (28 GHz)
        R = 6371e3  # Earth's radius in meters
        (lat1, lon1, lat2, lon2, alt1, alt2)=(user_lat, user_lon, sat_lat, sat_lon, user_alt, sat_alt)
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        ground_distance = R * c  # Distance over Earth's surface
        distance = np.sqrt(ground_distance**2 + (alt2 - alt1)**2)  # 3D distance considering altitude
        c = 3e8  # Speed of light in m/s
        fspl = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)   
           # Convert gains from dB to linear scale
        tx_gain_linear = 10**(tx_gain / 10)
        rx_gain_linear = 10**(rx_gain / 10)
   
    # Convert FSPL from dB to linear scale
        fspl_linear = 10**(fspl / 10)
        
    # Calculate the channel coefficient (linear scale)
        channel_coeff = torch.sqrt(torch.tensor(tx_gain_linear * rx_gain_linear / fspl_linear))
        # Rayleigh fading channel coefficients
        h = torch.sqrt(torch.tensor(1/2.0)) * torch.randn_like(X) + 1j * torch.randn_like(X)
        
        # Apply the Rayleigh fading channel

        X_leo = channel_coeff * h * X 
        X_leo = self.awgn(X_leo)
     # Assume MMSE signal recovery   
        X_leo = X_leo / channel_coeff 
        return X_leo        
    def build(self):
        constellation = torch.ones(self.M, 2)
        for i in range(self.M):
            constellation[i,0] = np.cos(2*np.pi*i/self.M)
            constellation[i,1] = np.sin(2*np.pi*i/self.M)
        return constellation
    
    def modulate(self, z:torch.Tensor):
        m = z.shape[0]
        X = torch.ones(int(m), 2)
        for i in range(m):
            X[i] = self.constellation[int(z[i])]
        return X
    
    def demodulate(self, X):
        inner = np.matmul(X, self.constellation.T)
        return np.argmax(inner, axis=1)


    
def ser(p, d):
    N= 2*d**2
    return 1.5*math.erfc(math.sqrt(p/(10*N)))



class QAM:
    def __init__(self, M, PSNR):
        self.M = M
        self.max = int(np.sqrt(self.M))-1
        self.constellation, self.map = self.build()
        self.p = (M-1)/6
        self.delta = self.compute_noise(PSNR)
        
    def compute_noise(self, PSNR):
        delta_2 = self.p/torch.pow(torch.tensor(10), PSNR/10).float()  
        return torch.sqrt(delta_2/2)
        
    def build(self):
        l = []
        d = {}
        m = int(np.sqrt(self.M))
        for i in range(m):
            for j in range(m):
                l.append((i,j))
                
        for i in range(self.M):
            d[l[i]] = i
        return l, d
    
    def modulate(self, z:torch.Tensor):
        m = z.shape[0]
        # print(m)
        X = torch.ones(int(m), 2)
        for i in range(m):
            x, y = self.constellation[int(z[i])]
            X[i,0] = x
            X[i,1] = y 
        return X
    
    def awgn(self, X):
        X += self.delta*torch.randn_like(X)
        return X
    def demodulate(self, X):
        m = X.shape[0]
        Z = torch.ones(m).long()
        for i in range(m):
            x = self.assign(X[i,0])
            y = self.assign(X[i,1])
            Z[i] = self.map[(x,y)]
        return Z
    
    def assign(self, ele):
        num = int(torch.round(ele))
        if num > self.max:
            num = self.max
        if num < 0:
            num = 0
        return num


