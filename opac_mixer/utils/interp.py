import numba
import numpy as np


@numba.njit(nogil=True, fastmath=True, cache=True)
def interp_2d(
    temp_old,
    press_old,
    temp_new,
    press_new,
    kcoeff,
    ls,
    lf,
    lg,
    lt_old,
    lp_old,
    lt_new,
    lp_new,
):
    """Function that interpolates to correct pressure and temperature."""
    kcoeff_new = np.empty((ls, lp_new, lt_new, lf, lg), dtype=np.float64)

    for speci in range(ls):
        to_i = temp_old[speci, : lt_old[speci]]
        po_i = press_old[speci, : lp_old[speci]]
        kcoeff_i = kcoeff[speci, : lp_old[speci], : lt_old[speci]]

        for gi in range(lg):
            for freqi in range(lf):
                # reset temporary array
                p_interp = np.empty((lp_new, lt_old[speci]), dtype=np.float64)
                pt_interp = np.empty((lp_new, lt_new), dtype=np.float64)

                # interp to new pressure (for all temperatures)
                for ti in range(lt_old[speci]):
                    p_interp[:, ti] = np.interp(
                        press_new, po_i, kcoeff_i[:, ti, freqi, gi]
                    )
                # interp to new temperature (for all -new- pressures)
                for pi in range(lp_new):
                    pt_interp[pi, :] = np.interp(temp_new, to_i, p_interp[pi, :])

                # Do edges
                for pi in range(lp_new):
                    for ti in range(lt_new):
                        if press_new[pi] < min(po_i) and temp_new[ti] < min(to_i):
                            pt_interp[pi, ti] = kcoeff[
                                speci, np.argmin(po_i), np.argmin(to_i), freqi, gi
                            ]
                        elif press_new[pi] < min(po_i) and temp_new[ti] > max(to_i):
                            pt_interp[pi, ti] = kcoeff[
                                speci, np.argmin(po_i), np.argmax(to_i), freqi, gi
                            ]
                        elif press_new[pi] > max(po_i) and temp_new[ti] < min(to_i):
                            pt_interp[pi, ti] = kcoeff[
                                speci, np.argmax(po_i), np.argmin(to_i), freqi, gi
                            ]
                        elif press_new[pi] > max(po_i) and temp_new[ti] > max(to_i):
                            pt_interp[pi, ti] = kcoeff[
                                speci, np.argmax(po_i), np.argmax(to_i), freqi, gi
                            ]

                kcoeff_new[speci, :, :, freqi, gi] = pt_interp

    return kcoeff_new
