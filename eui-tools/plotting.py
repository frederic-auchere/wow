from astropy.time import Time
import numpy as np


def draw_clock(date, ax, size=0.05):
    ax = ax.inset_axes([0, 0, size, size], projection='polar')
    for sp in ax.spines:
        ax.spines[sp].set_color('white')
    _, _, _, h, m, s = Time(date).to_value('ymdhms')
    if h > 12:
        h -= 12
    h = 2*np.pi*(1 - (h/12 + m/60 + s/3600))
    ax.plot((0, h), (0, 1), color='white')
    ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_theta_zero_location('N')
    ax.patch.set_facecolor('none')
