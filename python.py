import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=200):
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.x, self.y, self.z = self._generate_mesh()

    def _generate_mesh(self):
        U, V = self.U, self.V
        x = (self.R + V * np.cos(U / 2)) * np.cos(U)
        y = (self.R + V * np.cos(U / 2)) * np.sin(U)
        z = V * np.sin(U / 2)
        return x, y, z

    def surface_area(self):
        du = 2 * np.pi / self.n
        dv = self.w / self.n

        Xu = np.gradient(self.x, du, axis=1)
        Yu = np.gradient(self.y, du, axis=1)
        Zu = np.gradient(self.z, du, axis=1)
        Xv = np.gradient(self.x, dv, axis=0)
        Yv = np.gradient(self.y, dv, axis=0)
        Zv = np.gradient(self.z, dv, axis=0)

        cross = np.sqrt((Yu * Zv - Zu * Yv)**2 +
                        (Zu * Xv - Xu * Zv)**2 +
                        (Xu * Yv - Yu * Xv)**2)

        area = simpson(simpson(cross, self.v, axis=0), self.u)
        return area

    def edge_length(self):
        u = self.u
        v_edge = self.w / 2
        x1 = (self.R + v_edge * np.cos(u / 2)) * np.cos(u)
        y1 = (self.R + v_edge * np.cos(u / 2)) * np.sin(u)
        z1 = v_edge * np.sin(u / 2)

        x2 = (self.R - v_edge * np.cos(u / 2)) * np.cos(u)
        y2 = (self.R - v_edge * np.cos(u / 2)) * np.sin(u)
        z2 = -v_edge * np.sin(u / 2)

        p1 = np.vstack((x1, y1, z1)).T
        p2 = np.vstack((x2, y2, z2)).T

        def curve_length(points):
            diffs = np.diff(points, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            return np.sum(dists)

        return curve_length(p1) + curve_length(p2)

    def plot(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, rstride=4, cstride=4,
                        color='skyblue', edgecolor='k')
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()
strip = MobiusStrip(R=1.0, w=0.3, n=300)
area = strip.surface_area()
length = strip.edge_length()
strip.plot()

print("Surface Area:", area)
print("Edge Length:", length)
