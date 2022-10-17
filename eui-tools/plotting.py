from astropy.time import Time
import numpy as np


def draw_clock(date, ax, size=0.05):
    _, _, _, h, m, s = Time(date).to_value('ymdhms')
    if h > 12:
        h -= 12
    h = 2*np.pi*(h + m/60 + s/3600)/12
    m = 2*np.pi*(m/60 + s/3600)

    ax = ax.inset_axes([0, 0, size, size], projection='polar')
    ax.plot((0, h), (0, 0.5), color='white')
    ax.plot((0, m), (0, 1), color='white')
    ax.scatter(np.linspace(0, 2 * np.pi, 12, endpoint=False), np.ones(12), color='white', s=1)

    for sp in ax.spines.values():
        sp.set_color('white')
        sp.set_visible(False)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.patch.set_facecolor('none')
    ax.grid(False)
    ax.set_rmax(1.05)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
