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
from dataclasses import dataclass, field
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

# scale length dimensions to prevent overflow
LENGTH_SCALE_MULT = 1_000_000

# for cutoff envelope and first guess in non-linear regression
MAX_EPV = 7e22  # J
TEST_D = 20_000  # m
MAX_ITERATIONS = 500

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
    beta1_deg: float,
    beta2_deg: float,
    slope_deg: float,
    paleo_azimuth_uncertainty: float
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

def nan_if_untiltable(row):

    if is_tiltable(row['tilt'], row['dist']):
        return row['tilt']
    else:
        return np.nan

# PRELIMINARY TESTS


def tilt_check(
        test_az1: float,
        test_az2: float,
        test_sl2: float,
        paleo_azimuth_uncertainty: float = 7,
        max_abs_tilt: float = 5,
        plot_resolution: float = 3,
        plot_size: float = 5,
        plot_radius: float = 0.1,
        legend_position: tuple = (0.5, 0.5)
    ):  # type: ignore

    test_bearing = np.arange(360 * plot_resolution) / plot_resolution

    test_rim = 0.9 * plot_radius * np.ones(len(test_bearing))

    test_tilt = []

    for bearing in test_bearing:
        this_beta1 = signed_angular_difference(
            ang2_deg=test_az1, ang1_deg=bearing)
        this_beta2 = signed_angular_difference(
            ang2_deg=test_az2, ang1_deg=bearing)
        this_tilt = minimum_tilt(
            beta1_deg=this_beta1,
            beta2_deg=this_beta2,
            slope_deg=test_sl2, paleo_azimuth_uncertainty=paleo_azimuth_uncertainty
        )
        if np.abs(this_tilt) < max_abs_tilt:
            test_tilt.append(this_tilt)
        else:
            test_tilt.append(np.nan)

    nptest_tilt = np.array(test_tilt)

    f, ax = plt.subplots(
        subplot_kw={'projection': 'polar'}, figsize=(plot_size, plot_size), dpi=400)

    ax.set_theta_direction(-1)  # type: ignore
    ax.set_theta_offset(np.pi/2.0)  # type: ignore

    # radial limit
    plt.ylim(0, plot_radius)

    # plot az1 uncertainty range
    ax.vlines(
        x=np.radians((
            test_az1 + paleo_azimuth_uncertainty,
            test_az1 - paleo_azimuth_uncertainty
        )),
        ymin=0,
        ymax=plot_radius,
        label=f"$\\theta_1 \pm$" + f"{paleo_azimuth_uncertainty}",
        color='red'
    )

    # plot measured az1
    ax.vlines(x=np.radians(test_az1), ymin=0, ymax=plot_radius,
              colors=['black'], label=f"$\\theta_1=$" + f"{test_az1}")

    # plot modern topographic attitude
    sns.scatterplot(x=[np.radians(test_az2)], y=[np.sin(np.radians(test_sl2))],
                    label='$\\theta_2=$' + f'{test_az2}, \n' + '$\\varphi_2=$' + f'{test_sl2}', color='black', s=100)

    # nice radial ticks
    ax.set_yticks(np.linspace(0, plot_radius, 3)[1:])

    # plot calculated tilt by color around the rim. bearing + 180 to show direction TOWARD the center
    for i in np.linspace(0.9, 1.1, 20):
        if i == 0.9:
            sns.scatterplot(
                x=np.radians(test_bearing + 180),
                y=test_rim,
                hue=nptest_tilt,
                linewidth=0,
                palette='RdYlGn',
                size=1,
                # legend=False
            )
        else:
            sns.scatterplot(
                x=np.radians(test_bearing + 180),
                y=i * test_rim,
                hue=nptest_tilt,
                linewidth=0,
                palette='RdYlGn',
                size=1,
                legend=False  # type: ignore
            )

    # plt.title("Tilt required to explain a discordant sample \n by direction toward the center. All angles in degrees.")
    # plt.legend(, loc='upper left')
    handles, labels = ax.get_legend_handles_labels()

    # this is a hack to remove the "1" that appears randomly in the legend in the for loop implementation of thick rim
    handles = handles[:-1]
    labels = labels[:-1]

    ax.legend(
        handles[::-1],
        labels[::-1],
        loc='center',
        framealpha=1,
        bbox_to_anchor=legend_position
    )

    ax.set_axisbelow(True)


# CLASSES ______________

def plot_envelope(max_dist_km: float = 100_000, has_label: bool = True, color: str = 'black'):
    dist_m = np.arange(max_dist_km)
    max_tilt = mogi_tilt(dist_m, MAX_EPV, TEST_D)
    min_tilt = mogi_tilt(dist_m, -MAX_EPV, TEST_D)
    dist_km = dist_m / 1000

    if has_label:
        label = 'log$|E_{PV}\ /\ J|: $' + f'{np.round(np.log10(MAX_EPV), 1)}, ' + '$d: $' + f'{TEST_D/1000} km'
    else:
        label = None

    sns.lineplot(x=dist_km,y=max_tilt, c=color, label=label)
    sns.lineplot(x=dist_km,y=min_tilt, c=color)

def plot_tilt_distance_dataset(df: pd.DataFrame, color=None, name: str = ''):

    sns.scatterplot(
        x=df['dist_km'],
        y=np.zeros(len(df)),
        marker='|',
        s=200,
        color=color,
    )

    sns.scatterplot(
        data=df,
        x='dist_km',
        y='tilt',
        color=color,
        label=name
    )


@dataclass
class Population:
    name: str
    sIDs: list


@dataclass
class Center:
    cID: int
    lat: float
    lon: float
    data: pd.DataFrame
    paleo_azimuth_uncertainty: float = 7

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
                paleo_azimuth_uncertainty=self.paleo_azimuth_uncertainty
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

    def plot_tilt(
        self,
        pops: list[Population],
        exclude_untiltable: bool = True
    ):

        colors = itertools.cycle(sns.color_palette())  # type: ignore

        for pop in pops:

            next_color = next(colors)

            spot_check = self.get_data_subset(pop.sIDs)
            if exclude_untiltable:
                spot_check['tilt'] = spot_check.apply(nan_if_untiltable, axis=1)

            plot_tilt_distance_dataset(
                spot_check,
                next_color,
                name=f"Pop: {pop.name}"
            )


def make_center(cID, centers, samples, paleo_azimuth_uncertainty):
    return Center(
        cID=cID,
        lat=centers.loc[cID, 'LAT'],
        lon=centers.loc[cID, 'LON'],
        data=samples.copy(),
        paleo_azimuth_uncertainty=paleo_azimuth_uncertainty)


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

def frac_tiltable(df: pd.DataFrame) -> dict:
    score = {'frac_tiltable': np.nan}
    try:
        num_samples = len(df)
        num_tiltable = len(df[df['tilt'].notnull()])
        score['frac_tiltable'] = num_tiltable / num_samples
    except:
        pass
    return score

def inflation_score(df: pd.DataFrame) -> dict:

    # df_subset = df.loc[df['is_tiltable']]
    df_subset = df.loc[df['tilt'].notnull()] # no envelope version

    # initialize output
    scores = {
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
class Criterion:
    func: Callable
    pop: Population


def evaluate_center(center: Center, crit: Criterion):
    pop_subset = center.data.loc[crit.pop.sIDs]
    return crit.func(pop_subset)


# MODEL - RELATED CLASSES -------

@dataclass
class Node:
    pos1: tuple
    disp: tuple

    def __post_init__(self):
        self.pos2 = self.pos1 + self.disp


@dataclass
class Edge:
    proximal: Node  # (A in text)
    distal: Node  # (B in text)

    def __post_init__(self):
        # relative dimensions [r, z] of initial and displaced segments
        self.shape1 = self.distal.pos1 - self.proximal.pos1  # type: ignore
        self.shape2 = self.distal.pos2 - self.proximal.pos2  # type: ignore

        # mean position of initial and displaced segments
        self.pos1 = (self.distal.pos1 + self.proximal.pos1) / 2  # type: ignore
        self.pos2 = (self.distal.pos2 + self.proximal.pos2) / 2  # type: ignore
        self.disp = (self.distal.disp + self.proximal.disp) / 2  # type: ignore

        self.disp_r = self.disp[0]
        self.disp_z = self.disp[1]

        # radial distances for plotting
        self.dist = self.pos2[0]  # type: ignore
        self.dist_km = self.dist / 1000

        # initial and displaced slopes (positive downward from center)
        # index [1] is z component; [0] is r component
        self.slope1 = deg(arctan2(-self.shape1[1], self.shape1[0]))
        self.slope2 = deg(arctan2(-self.shape2[1], self.shape2[0]))

        self.tilt = self.slope2 - self.slope1


model_path = "../GEOL192-Model/data/"


def set_model_path(path: str):
    model_path = str


# paleo-edifice spline data
topo = np.genfromtxt(f'{model_path}z1.csv', delimiter=",").T


def model_pos1_from_csv(name: str):
    r = np.genfromtxt(model_path + "rdisp_" + name, delimiter=",")[:, 0]
    z = np.interp(r, *topo, right=0)  # interpolate z1 into topography
    return np.array([r, z]).T


def model_disp_from_csv(name: str):
    r = np.genfromtxt(f'{model_path}rdisp_{name}', delimiter=",")[:, 1]
    z = np.genfromtxt(f'{model_path}zdisp_{name}', delimiter=",")[:, 1]
    return np.array([r, z]).T


def read_model_data(params: dict):

    filename = f"depth_{params['depth']}_radius_{params['radius']}_aspect_{params['aspect']}_pmult_{params['pmult']}_grav_{int(params['grav'])}_topo_{int(params['topo'])}.csv"

    pos1 = model_pos1_from_csv(filename)
    disp = model_disp_from_csv(filename)

    # make z1 flat for flat model
    if not params['topo']:
        pos1[1] = np.zeros(len(pos1[1]))

    # subtract out gravitational component (from no overpressure)
    if params['grav']:
        filename_p0 = filename.replace(
            f"pmult_{params['pmult']}", "pmult_0")
        disp -= model_disp_from_csv(filename_p0)

    return {'disp': disp, 'pos1': pos1}


@dataclass
class NumericalModel:
    params: dict
    pos1: np.array = field(repr=False)  # type: ignore
    disp: np.array = field(repr=False) # type: ignore
    rmse: float = np.nan

    def __post_init__(self):

        self.set_params()
        self.build_nodes()
        self.build_edges()

    def set_params(self):
        self.radius = self.params['radius']
        self.half_height = self.radius * self.params['aspect']

        self.over_pressure = self.params['pmult'] * \
            self.params['depth'] * ROCK_DENSITY * MARS_GRAVITY

        self.res_vol = (4 / 3) * np.pi * self.radius**2 * self.half_height

        self.epv = self.over_pressure * self.res_vol
        self.depth = self.params['depth'] + self.half_height

    def build_nodes(self):
        self.nodes = [
            Node(self.pos1[i], self.disp[i]) for i in range(len(self.pos1))
        ]

    def build_edges(self):
        self.consecutive_node_pairs = zip(self.nodes[:-1], self.nodes[1:])
        self.edges = [
            Edge(*pair) for pair in self.consecutive_node_pairs
        ]

        # put edge attributes into dict of lists
        self.data = pd.DataFrame(
            [vars(edge) for edge in self.edges]
        )

        # self.attributes = self.data.to_dict("list")

    def plot_numerical_tilt(self):
        sns.lineplot(data=self.data, x='dist_km', y='tilt')

    def plot_numerical_displacement(self):
        sns.lineplot(data=self.data, x='dist_km', y='disp_r', label='r')
        sns.lineplot(data=self.data, x='dist_km', y='disp_z', label='z')


def unpack_param_combinations(dict_of_lists):
    keys = dict_of_lists.keys()
    all_vals = list(itertools.product(*dict_of_lists.values()))
    list_of_dicts = [dict(zip(keys, vals)) for vals in all_vals]
    return list_of_dicts


def make_numerical_model(params: dict):
    model = NumericalModel(
        params=params,
        pos1=read_model_data(params)['pos1'],
        disp=read_model_data(params)['disp']
    )
    return model


def numerical_model_rmse(
        model: NumericalModel,
        map_tilt_df: pd.DataFrame
    ) -> float:

    possible_subset = map_tilt_df[map_tilt_df['tilt'].notnull()]

    tilt_map = possible_subset['tilt'].tolist()
    dist_km_map = possible_subset['dist_km'].tolist()

    tilt_numerical_model = model.data['tilt'].tolist()
    dist_km_numerical_model = model.data['dist_km'].tolist()

    tilt_predicted = np.interp(
        x=dist_km_map, # type: ignore
        xp=dist_km_numerical_model,
        fp=tilt_numerical_model
    )

    rmse = np.sqrt(mean_squared_error(
        y_true=tilt_map,
        y_pred=tilt_predicted
    ))

    return rmse # type: ignore


@dataclass
class ParamSweep:
    all_params: list[dict]

    def __post_init__(self):

        self.models = [
            make_numerical_model(params) for params in self.all_params
        ]

    def sort_models_by_rmse(
        self,
        center: Center,
        pops: list[Population],
        tiltable_only: bool = True,
    ) -> None:

        sIDs = []
        for pop in pops:
            for sID in pop.sIDs:
                sIDs.append(sID)

        data = center.data.loc[sIDs]

        if tiltable_only:
            data['tilt'] = data.apply(nan_if_untiltable, axis=1)

        data_no_nulls = data[data['tilt'].notnull()]

        for model in self.models:
            try:
                model.rmse = numerical_model_rmse(model, data_no_nulls) # type: ignore
            except:
                pass

        self.models.sort(key=lambda model: model.rmse)