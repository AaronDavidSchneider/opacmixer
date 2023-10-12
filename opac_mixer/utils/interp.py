"""Module that houses the 2D interpolation routine"""
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
    """
    Function that does a bilinear interpolation on k-tables in 2D to correct pressure and temperature.
    Numba accelerated.

    Parameters
    ----------
    temp_old (array(lt_old)):
        the temperature values of the original grid
    press_old (array(lp_old)):
        the pressure values of the original grid
    temp_new (array(lt_new)):
        the temperature values to which we interpolate
    press_new (array(lp_new)):
        the pressure values to which we interpolate
    kcoeff: array(ls, lp_old, lt_old, lf, lg
        the k-table which is to be interpolated to new pressure temperature values
    ls: int
        Number of opacity species in k-table grid
    lf: int
        Number of frequency points in k-table grid
    lg: int
        Number of $g$ values in k-table grid
    lt_old: int
        number of temperature points in original grid
    lp_old: int
        number of pressure points in original grid
    lt_new: int
        number of temperature points to which we interpolate
    lp_new: int
        number of pressure points to which we interpolate

    Returns
    -------
    kcoeff_new (array(ls, lp_new, lt_new, lf, lg)):
        The interpolated k-table grid
    """
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
                    pt_interp[pi, :] = np.interp(
                        temp_new, to_i, p_interp[pi, :]
                    )

                # Do edges
                for pi in range(lp_new):
                    for ti in range(lt_new):
                        if press_new[pi] < min(po_i) and temp_new[ti] < min(
                            to_i
                        ):
                            pt_interp[pi, ti] = kcoeff[
                                speci,
                                np.argmin(po_i),
                                np.argmin(to_i),
                                freqi,
                                gi,
                            ]
                        elif press_new[pi] < min(po_i) and temp_new[ti] > max(
                            to_i
                        ):
                            pt_interp[pi, ti] = kcoeff[
                                speci,
                                np.argmin(po_i),
                                np.argmax(to_i),
                                freqi,
                                gi,
                            ]
                        elif press_new[pi] > max(po_i) and temp_new[ti] < min(
                            to_i
                        ):
                            pt_interp[pi, ti] = kcoeff[
                                speci,
                                np.argmax(po_i),
                                np.argmin(to_i),
                                freqi,
                                gi,
                            ]
                        elif press_new[pi] > max(po_i) and temp_new[ti] > max(
                            to_i
                        ):
                            pt_interp[pi, ti] = kcoeff[
                                speci,
                                np.argmax(po_i),
                                np.argmax(to_i),
                                freqi,
                                gi,
                            ]

                kcoeff_new[speci, :, :, freqi, gi] = pt_interp

    return kcoeff_new
