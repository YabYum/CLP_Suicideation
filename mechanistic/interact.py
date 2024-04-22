
import numpy as np
import matplotlib.pyplot as plt
from helper import create_likelihood_matrix, get_prior, inference, collect, impulse, step
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class Simulate:

    def __init__(self, size, decay, weight, priobs, intfire, T):
        self.sizes, self.sizei, self.sized, self.sizep = size
        self.decays1, self.decays2, self.decayi1, self.decayi2, self.decayd1, self.decayd2, self.decayp1, self.decayp2 = decay
        self.w_si, self.w_sd, self.w_sp, self.w_id, self.w_ip, self.w_dp = weight
        self.pri_s, self.pri_i, self.pri_d, self.obs_s, self.obs_i, self.obs_d = priobs
        self.dt, self.tau, self.el, self.theta, self.t_step = intfire
        self.T = T
    
    def likeli_prios(self):
        likelihood_s, likelihood_i, likelihood_d = create_likelihood_matrix(self.sizes, self.decays1), create_likelihood_matrix(self.sizei, self.decayi1), create_likelihood_matrix(self.sized, self.decayd1)
        prior_s, prior_i, prior_d = get_prior(self.pri_s, self.sizes, self.decays2), get_prior(self.pri_i, self.sizei, self.decayi2), get_prior(self.pri_d, self.sized, self.decayd2)
        return likelihood_s, likelihood_i, likelihood_d, prior_s, prior_i, prior_d
    
    def run(self):
        likelihood_s, likelihood_i, likelihood_d, prior_s, prior_i, prior_d = self.likeli_prios()
        Fss_history, Fii_history, Fdd_history, Fp_history = [], [], [], []
        Fs_min_history, Fi_min_history, Fd_min_history = [], [], []
        qss_history, qsi_history, qsd_history = [], [], []
        v_rec, t_rec, s_rec = np.array([]), np.array([]), np.array([])

        qss, Fs = inference(likelihood=likelihood_s, prior=prior_s, obs=self.obs_s)
        Fss = Fs[self.obs_s]
        Fss_history.append(Fss)
        Fs_min = Fs[Fs.argmin()]
        Fs_min_history.append(Fs_min)
        qss_history.append(qss.argmax())

        qsi, Fi = inference(likelihood_i, prior_i, self.obs_i)
        qsi_history.append(qsi.argmax())
        Fii = Fi[self.obs_i]
        Fii_history.append(Fii)
        Fi_min = Fi[Fi.argmin()]
        Fi_min_history.append(Fi_min)

        qsd, Fd = inference(likelihood_d, prior_d, self.obs_d)
        qsd_history.append(qsd.argmax())
        Fdd = Fd[self.obs_d]
        Fdd_history.append(Fdd)
        Fd_min = Fd[Fd.argmin()]
        Fd_min_history.append(Fd_min)

        Fp = collect(Fss, Fii, Fdd, self.w_sp, self.w_ip, self.w_dp)
        Fp_history.append(Fp)

        v = self.el
        dt = 0.1
        for t in range (int(1/dt)):
            s, v, v_rec, t_rec, s_rec = impulse(v_rec, t_rec, s_rec, v, self.theta, dt, self.tau, self.el, Fp, t)

        for _ in range(self.T):
            qss, Fs, qsss, Fss = step(likelihood_s, prior_s, self.obs_s, Fii, self.w_si, Fdd, self.w_sd)
            Fss_history.append(Fss)
            Fs_min_history.append(Fs.min())
            qss_history.append(qsss)  # Assuming qsss is the updated posterior

            qsi, Fi, qsii, Fii = step(likelihood_i, prior_i, self.obs_i, Fss, self.w_si, Fdd, self.w_id)
            Fii_history.append(Fii)
            Fi_min_history.append(Fi.min())
            qsi_history.append(qsii)

            qsd, Fd, qsdd, Fdd = step(likelihood_d, prior_d, self.obs_d, Fss, self.w_sd, Fii, self.w_id)
            Fdd_history.append(Fdd)
            Fd_min_history.append(Fd.min())
            qsd_history.append(qsdd)
            Fp = collect(Fss, Fii, Fdd, self.w_sp, self.w_ip, self.w_dp)
            Fp_history.append(Fp)

            for t in range (int(1/dt)):
                s, v, v_rec, t_rec, s_rec = impulse(v_rec, t_rec, s_rec, v, self.theta, dt, self.tau, self.el, Fp, t)

        
        return Fss_history, Fii_history, Fdd_history, Fp_history, Fs_min_history, Fi_min_history, Fd_min_history, v_rec, s_rec


class Interact:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulating suicidal thoughts")
        
        ttk.Label(root, text="Symbolic ambiguity:").grid(column=0, row=0)
        self.decays2_slider = ttk.Scale(root, from_=0.1, to=5.0, orient='horizontal')
        self.decays2_slider.grid(column=1, row=0)
        self.decays2_slider.set(2)#default
        
        ttk.Label(root, text="Imaginary ambiguity:").grid(column=0, row=1)
        self.decayi2_slider = ttk.Scale(root, from_=0.1, to=5.0, orient='horizontal')
        self.decayi2_slider.grid(column=1, row=1)
        self.decayi2_slider.set(1)#default

        ttk.Label(root, text="Real ambiguity:").grid(column=0, row=2)
        self.decayd2_slider = ttk.Scale(root, from_=0.1, to=5.0, orient='horizontal')
        self.decayd2_slider.grid(column=1, row=2)
        self.decayd2_slider.set(2)#default

        ttk.Label(root, text="Symbolic prior:").grid(column=0, row=3)
        self.sympr_slider = ttk.Scale(root, from_=0, to=49, orient='horizontal')
        self.sympr_slider.grid(column=1, row=3)
        self.sympr_slider.set(25)  # default

        ttk.Label(root, text="Symbolic observation:").grid(column=0, row=4)
        self.symob_slider = ttk.Scale(root, from_=0, to=49, orient='horizontal')
        self.symob_slider.grid(column=1, row=4)
        self.symob_slider.set(25)#default

        ttk.Label(root, text="Imaginary prior:").grid(column=0, row=5)
        self.impr_slider = ttk.Scale(root, from_=0, to=49, orient='horizontal')
        self.impr_slider.grid(column=1, row=5)
        self.impr_slider.set(25)#default

        tk.Label(root, text="Imaginary observation:").grid(column=0, row=6)
        self.imob_slider = ttk.Scale(root, from_=0, to=49, orient='horizontal')
        self.imob_slider.grid(column=1, row=6)
        self.imob_slider.set(25)#default

        ttk.Label(root, text="Real prior:").grid(column=0, row=7)
        self.repr_slider = ttk.Scale(root, from_=0, to=49, orient='horizontal')
        self.repr_slider.grid(column=1, row=7)
        self.repr_slider.set(25)#default

        tk.Label(root, text="Real observation:").grid(column=0, row=8)
        self.reob_slider = ttk.Scale(root, from_=0, to=49, orient='horizontal')
        self.reob_slider.grid(column=1, row=8)
        self.reob_slider.set(25)#default

        tk.Label(root, text="Carrying capability:").grid(column=0, row=9)
        self.cacp_slider = ttk.Scale(root, from_=0, to=49, orient='horizontal')
        self.cacp_slider.grid(column=1, row=9)
        self.cacp_slider.set(20)#default

        self.run_button = ttk.Button(root, text="Run Simulation", command=self.run_simulation)
        self.run_button.grid(column=0, row=10, columnspan=2)
        
    def plot(self, Fss_history, Fii_history, Fdd_history, Fp_history, Fs_min_history, Fi_min_history, Fd_min_history, v_rec, s_rec):
        Fi_min_history1 = [x + 0.6 for x in Fi_min_history]
        dt = 0.1

        # Clear the previous figure if any
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

        ax1.plot(Fss_history, linestyle='-', color='black')
        ax1.plot(Fii_history, linestyle='-', color='green')
        ax1.plot(Fdd_history, linestyle='-', color='gray')
        ax1.plot(Fp_history, linestyle='-', color='red')
        ax1.plot(Fs_min_history, linestyle='--', color='black')
        ax1.plot(Fi_min_history1, linestyle='--', color='green')
        ax1.plot(Fd_min_history, linestyle='--', color='gray')
        ax1.set_ylabel('Free energy')

        ax2.plot(v_rec, color='black', linestyle='-.')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Accumulation')

        ax3.plot(s_rec, '.', markersize=20)
        ax3.axis([0, 21/dt, 0.8, 1.2])
        ax3.set_ylabel('Impulse')

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=11, columnspan=3)

    def run_simulation(self):
        decays2 = self.decays2_slider.get()
        decayi2 = self.decays2_slider.get()
        decayd2 = self.decays2_slider.get()
        sympr = int(self.sympr_slider.get())
        symob = int(self.symob_slider.get())
        impr = int(self.impr_slider.get())
        imob = int(self.imob_slider.get())
        repr = int(self.repr_slider.get())
        reob = int(self.reob_slider.get())
        cacp = int(self.cacp_slider.get())

        
        size = [50, 50, 50, 8]
        decay = [2, decays2, 2, decayi2, 2, decayd2, 2, 1]
        weight= [0.5, 0.1, 0.9, 0.2, 0.8, 0.7]
        priobs = [sympr, impr, repr, symob, imob, reob]
        intfire = [0.1, 10, 0, cacp, 0]
        T = 20
        
        sim = Simulate(size, decay, weight, priobs, intfire, T)
        Fss_history, Fii_history, Fdd_history, Fp_history, Fs_min_history, Fi_min_history, Fd_min_history, v_rec, s_rec = sim.run()
        self.plot(Fss_history, Fii_history, Fdd_history, Fp_history, Fs_min_history, Fi_min_history, Fd_min_history, v_rec, s_rec)

if __name__ == "__main__":
    root = tk.Tk()
    app = Interact(root)
    root.mainloop()
