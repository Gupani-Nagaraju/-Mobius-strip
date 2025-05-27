import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.3, n=200):
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self._generate_mesh()

    def _generate_mesh(self):
        u, v = self.U, self.V
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def compute_surface_area(self):
        # Partial derivatives
        dXdu = np.gradient(self.X, axis=1)
        dXdv = np.gradient(self.X, axis=0)
        dYdu = np.gradient(self.Y, axis=1)
        dYdv = np.gradient(self.Y, axis=0)
        dZdu = np.gradient(self.Z, axis=1)
        dZdv = np.gradient(self.Z, axis=0)

        # Cross product of partials
        N1 = dYdu * dZdv - dZdu * dYdv
        N2 = dZdu * dXdv - dXdu * dZdv
        N3 = dXdu * dYdv - dYdu * dXdv

        dA = np.sqrt(N1**2 + N2**2 + N3**2)

        # Integrate
        surface_area = simpson(simpson(dA, self.v), self.u)
        return surface_area

    def compute_edge_length(self):
        edge_v = np.full_like(self.u, self.w / 2)
        x_edge = (self.R + edge_v * np.cos(self.u / 2)) * np.cos(self.u)
        y_edge = (self.R + edge_v * np.cos(self.u / 2)) * np.sin(self.u)
        z_edge = edge_v * np.sin(self.u / 2)

        dx = np.gradient(x_edge)
        dy = np.gradient(y_edge)
        dz = np.gradient(z_edge)
        ds = np.sqrt(dx**2 + dy**2 + dz**2)

        return np.sum(ds)

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=4, cstride=4, color='teal', alpha=0.8, edgecolor='k')
        ax.set_title("Möbius Strip")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.4, n=300)
    print(f"Surface Area ≈ {mobius.compute_surface_area():.4f}")
    print(f"Edge Length ≈ {mobius.compute_edge_length():.4f}")
    mobius.plot()
