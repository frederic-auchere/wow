from astropy.time import Time
import numpy as np
from matplotlib import patches


def draw_clock(date, ax, size=0.05):
    _, _, _, h, m, s = Time(date).to_value('ymdhms')
    if h > 12:
        h -= 12
    hours = 2*np.pi*(h + m/60 + s/3600)/12
    minutes = 2*np.pi*(m/60 + s/3600)

    ax = ax.inset_axes([0, 0, size, size], projection='polar')
    hours_hand, = ax.plot((0, hours), (0, 0.5), color='white')
    hours_hand.set_solid_capstyle('round')
    minutes_hand, = ax.plot((0, minutes), (0, 1), color='white')
    minutes_hand.set_solid_capstyle('round')
    ax.scatter(np.linspace(0, 2*np.pi, 12, endpoint=False), np.ones(12), color='white', s=1)

    for sp in ax.spines.values():
        sp.set_color('white')
        sp.set_visible(False)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.patch.set_alpha(0.25)


def make_subplot(image, ax, norm,
                 title=None, x_lims=None, y_lims=None, interpolation='nearest', cmap='gray', inset=None, clock=None):
    ax.imshow(image, origin='lower', norm=norm, cmap=cmap, interpolation=interpolation)
    if x_lims is None:
        x_lims = 0, image.shape[1]
    if y_lims is None:
        y_lims = 0, image.shape[0]
    foot_position = 0
    if clock:
        size = 0.05
        draw_clock(clock, ax, size=size)
        foot_position += size + 0.01
    if title:
        ax.text(foot_position, 0,
                title,
                transform=ax.transAxes,
                color='white',
                # bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.5, 'pad': 1},
                ha='left', va='bottom')
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    ax.axis(False)
    if inset:
        axins = ax.inset_axes([0, 0.5, 0.5, 0.5])
        axins.imshow(image, norm=norm, origin='lower', cmap=cmap, interpolation=interpolation)
        inset_x_lims, inset_y_lims = (inset[0], inset[1]), (inset[2], inset[3])
        axins.set_xlim(*inset_x_lims)
        axins.set_ylim(*inset_y_lims)
        axins.axis('off')
        x_mid = (x_lims[1] - x_lims[0])/2
        y_mid = (y_lims[1] - y_lims[0])/2
        ax.plot([0, x_mid], [y_mid, y_mid], '-', color='white')
        ax.plot([x_mid, x_mid], [y_mid, y_lims[1]], '-', color='white')
        rect = patches.Rectangle((inset[0], inset[2]), inset[1] - inset[0], inset[3] - inset[2],
                                 linewidth=1, edgecolor='white', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
