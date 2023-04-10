import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from collections.abc import Iterable
from typing import Callable, Tuple
import numpy as np
from numpy import radians as rad
from numpy import degrees as deg
from numpy import sin, cos, tan, arcsin, arccos, arctan, arctan2, mean

import pandas as pd
from dataclasses import dataclass
import itertools

import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import mean_squared_error

warnings.simplefilter("error", OptimizeWarning)


SHEAR_MODULUS = 2.4e10  # Pa
ROCK_DENSITY = 2700  # kg / m^3
MAGMA_DENSITY = 2700  # kg / m^3
MARS_GRAVITY = 3.72  # m / s^2
MARS_EQ_RADIUS = 3_396_200  # m

AZ1_UNCERTAINTY = 7  # degrees

# for plotting numerical
PLOT_WIDTH = 200_000

# scale length dimensions to prevent overflow
LENGTH_SCALE_MULT = 1_000_000

# for cutoff envelope and first guess in non-linear regression
MAX_EPV = 7e22  # J
TEST_D = 20_000  # m
MAX_ITERATIONS = 80

# parameter conversion factor
EPV_OVER_K = 16 * np.pi * SHEAR_MODULUS / 9


# FUNCTIONS ____________


def mogi_tilt(dist, epv, d, length_scale_mult=LENGTH_SCALE_MULT):
    k = epv / EPV_OVER_K

    # scale to prevent overflow
    r1_scale = dist / length_scale_mult
    k_scale = k / (length_scale_mult ** 3)
    d_scale = d / length_scale_mult

    num = 3 * k_scale * d_scale * r1_scale
    denom = (d_scale**2 + r1_scale**2)**2.5 + \
        k_scale * (d_scale**2 - 2*r1_scale**2)
    return np.degrees(np.arctan2(num, denom))


def epv_numerical_model(depth, radius, aspect, pmult):

    volume = (4/3) * np.pi * radius**3 * aspect
    pressure = ROCK_DENSITY * MARS_GRAVITY * depth * pmult
    return volume * pressure


def great_circle_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg, radius=MARS_EQ_RADIUS):
    '''distance between pts 1 and 2'''
    lat1 = rad(lat1_deg)
    lon1 = rad(lon1_deg)
    lat2 = rad(lat2_deg)
    lon2 = rad(lon2_deg)

    angular_distance = arccos(
        cos(lat1) * cos(lat2)
        * cos(lon2 - lon1)
        + sin(lat1) * sin(lat2)
    )

    return angular_distance * radius


def great_circle_bearing(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    '''azimuth from pt 1 AWAY from pt 2'''

    lat1 = rad(lat1_deg)
    lon1 = rad(lon1_deg)
    lat2 = rad(lat2_deg)
    lon2 = rad(lon2_deg)

    y = sin(lon2 - lon1) * cos(lat2)
    x = cos(lat1) * sin(lat2) \
        - sin(lat1) * cos(lat2) * cos(lon2 - lon1)

    return (180 + deg(arctan2(y, x))) % 360


def signed_angular_difference(ang2_deg, ang1_deg):
    '''angular difference expressed in range -180 to 180'''

    return ((ang2_deg - ang1_deg + 180) % 360) - 180


def angular_difference(ang2_deg, ang1_deg):
    return np.abs(signed_angular_difference(ang2_deg, ang1_deg))


def paleo_slope(beta1_deg, beta2_deg, sl2_deg):
    beta1 = rad(beta1_deg)
    beta2 = rad(beta2_deg)
    sl2 = rad(sl2_deg)
    argument = sin(beta2) * sin(sl2) / sin(beta1)
    if argument < 0 or argument > 1:  # not possible
        return np.nan
    else:
        return deg(arcsin(argument))


def great_circle_projection(beta_deg, slope_deg):
    '''project point onto great circle'''

    beta = rad(beta_deg)
    slope = rad(slope_deg)
    proj = arctan(tan(slope) * cos(beta))
    return deg(proj)


def minimum_tilt(
    beta1_deg,
    beta2_deg,
    slope_deg,
    paleo_azimuth_uncertainty=AZ1_UNCERTAINTY
):

    # already within uncertainty -> tilt = 0
    if np.abs(signed_angular_difference(
            beta1_deg, beta2_deg
    )) < paleo_azimuth_uncertainty:
        return 0

    # beta1 uncertainty boundaries
    beta1_possible = np.array([
        beta1_deg + paleo_azimuth_uncertainty,
        beta1_deg - paleo_azimuth_uncertainty
    ])

    nearest_beta1 = min(
        beta1_possible,
        key=lambda beta1_deg: angular_difference(beta1_deg, beta2_deg)
    )

    best_paleo_slope = paleo_slope(
        beta1_deg=nearest_beta1,
        beta2_deg=beta2_deg,
        sl2_deg=slope_deg
    )

    # no paleo_slope -> no tilt
    if np.isnan(best_paleo_slope):
        return np.nan

    # evaluate tilt
    tilt = great_circle_projection(beta2_deg, slope_deg) \
        - great_circle_projection(nearest_beta1, best_paleo_slope)

    return tilt


def is_tiltable(tilt, dist_m):
    '''check whether a given tilt-distance pair is within the tilt envelope'''

    if np.isnan(tilt):
        return False

    max_tilt = mogi_tilt(dist=dist_m, epv=MAX_EPV, d=TEST_D)
    min_tilt = mogi_tilt(dist=dist_m, epv=-MAX_EPV, d=TEST_D)

    if tilt > max_tilt:
        return False
    if tilt < min_tilt:
        return False

    return True


# CLASSES ______________

@dataclass
class Center:
    cID: int
    lat: float
    lon: float
    data: pd.DataFrame

    def __post_init__(self) -> None:

        # this of course only works for unique self.cIDs
        self.calculate()

    def get_data_subset(self, sIDs: list) -> pd.DataFrame:
        return self.data.loc[sIDs]

    def calculate(self):

        self.data['dist'] = self.data.apply(
            lambda row: great_circle_distance(
                lat1_deg=row['LAT'], lon1_deg=row['LON'],
                lat2_deg=self.lat, lon2_deg=self.lon
            ),
            axis=1
        )

        self.data['dist_km'] = self.data.apply(
            lambda row: row['dist'] / 1000,
            axis=1
        )

        self.data['bearing'] = self.data.apply(
            lambda row: great_circle_bearing(
                lat1_deg=row['LAT'], lon1_deg=row['LON'],
                lat2_deg=self.lat, lon2_deg=self.lon
            ),
            axis=1
        )

        self.data['beta1'] = self.data.apply(
            lambda row: signed_angular_difference(
                ang2_deg=row['AZ1'], ang1_deg=row['bearing']
            ),
            axis=1
        )

        self.data['beta2'] = self.data.apply(
            lambda row: signed_angular_difference(
                ang2_deg=row['AZ2'], ang1_deg=row['bearing']
            ),
            axis=1
        )

        self.data['tilt'] = self.data.apply(
            lambda row: minimum_tilt(
                beta1_deg=row['beta1'],
                beta2_deg=row['beta2'],
                slope_deg=row['SL2'],
            ),
            axis=1
        )

        self.data['is_tiltable'] = self.data.apply(
            lambda row: is_tiltable(
                tilt=row['tilt'],
                dist_m=row['dist']
            ),
            axis=1
        )


def make_center(cID, centers, samples):
    return Center(
        cID=cID,
        lat=centers.loc[cID, 'LAT'],
        lon=centers.loc[cID, 'LON'],
        data=samples.copy())


def summit_score(df: pd.DataFrame) -> dict:
    try:
        score = {'summit_score': mean(np.abs(df['beta1']))}
    except:
        score = {'summit_score': np.nan}
    return score


def fit_mogi_function(df: pd.DataFrame, full_output: bool = False):

    # initial guess lower than envelope
    if np.mean(df['tilt']) < 0:
        p0 = -MAX_EPV / 1000, TEST_D
    else:
        p0 = MAX_EPV / 1000, TEST_D

    fit = curve_fit(
        f=mogi_tilt,
        xdata=df['dist'],
        ydata=df['tilt'],
        p0=p0,
        maxfev=MAX_ITERATIONS,
        method='lm',
        full_output=full_output
    )

    return fit


def inflation_score(df: pd.DataFrame) -> dict:

    # get size of population before taking tiltable subset
    pop_size = len(df)

    df_subset = df.loc[df['is_tiltable']]

    # initialize output
    scores = {
        # doesn't depend on fit results
        'frac_tiltable': len(df_subset) / pop_size,
        'log10_epv': np.nan,
        'epv_is_positive': np.nan,
        'depth': np.nan,
        'rmse': np.nan,
    }

    # attempt regression
    try:
        params, _, infodict, _, _ = fit_mogi_function(
            df_subset,
            full_output=True
        )

        # unpack param estimate and root mean squared error
        epv, depth = params
        rmse = np.sqrt(mean_squared_error(
            y_pred=infodict['fvec'],
            y_true=df_subset['tilt']
        ))

        # rewrite scores in dict
        scores['log10_epv'] = np.log10(np.abs(epv))
        scores['epv_is_positive'] = epv > 0  # boolean
        scores['depth'] = depth
        scores['rmse'] = rmse  # type: ignore

    # catch regression failure
    except OptimizeWarning:  # does not converge
        pass

    except RuntimeError:
        pass

    except ValueError:
        pass

    except TypeError:  # 'func input vector length N=2 must not exceed func output vector length M=1'
        pass

    return scores


@dataclass
class Population:
    name: str
    sIDs: list


@dataclass
class Criterion:
    func: Callable
    pop: Population


def evaluate_center(center: Center, crit: Criterion):
    pop_subset = center.data.loc[crit.pop.sIDs]
    return crit.func(pop_subset)
