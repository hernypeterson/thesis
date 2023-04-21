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

# physical properties
SHEAR_MODULUS_PA = 2.4e10
ROCK_DENSITY_KG_M3 = 2700
MARS_GRAVITY_M_S2 = 3.72
MARS_EQ_RADIUS_M = 3_396_200

# for non-linear regression
TEST_EPV_J = 7e19
TEST_DEPTH_M = 20_000
MAX_ITERATIONS = 500

# parameter conversion factor in mogi_tilt
EPV_OVER_K = 16 * np.pi * SHEAR_MODULUS_PA / 9
# scale length dimensions to prevent overflow in mogi_tilt
LENGTH_SCALE_MULT = 1_000_000


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


def epv_numerical_model(depth, radius, aspect, pmult) -> float:
    volume = (4/3) * np.pi * radius**3 * aspect
    pressure = ROCK_DENSITY_KG_M3 * MARS_GRAVITY_M_S2 * depth * pmult
    return volume * pressure


def great_circle_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg, radius=MARS_EQ_RADIUS_M) -> float:
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


def great_circle_bearing(lat1_deg, lon1_deg, lat2_deg, lon2_deg) -> float:
    '''azimuth from pt 1 AWAY from pt 2'''
    lat1 = rad(lat1_deg)
    lon1 = rad(lon1_deg)
    lat2 = rad(lat2_deg)
    lon2 = rad(lon2_deg)

    y = sin(lon2 - lon1) * cos(lat2)
    x = cos(lat1) * sin(lat2) \
        - sin(lat1) * cos(lat2) * cos(lon2 - lon1)

    return (180 + deg(arctan2(y, x))) % 360


def signed_angular_difference(ang2_deg, ang1_deg) -> float:
    '''angular difference expressed in range -180 to 180'''
    return ((ang2_deg - ang1_deg + 180) % 360) - 180


def angular_difference(ang2_deg, ang1_deg) -> float:
    return np.abs(signed_angular_difference(ang2_deg, ang1_deg))


def paleo_slope(beta1_deg, beta2_deg, sl2_deg):
    beta1 = rad(beta1_deg)
    beta2 = rad(beta2_deg)
    sl2 = rad(sl2_deg)
    argument = sin(beta2) * sin(sl2) / sin(beta1)
    if argument < 0 or argument > 1:
        return np.nan
    else:
        return deg(arcsin(argument))


def great_circle_projection(beta_deg, slope_deg):
    '''project point onto great circle'''
    beta = rad(beta_deg)
    slope = rad(slope_deg)
    proj = arctan(tan(slope) * cos(beta))
    return deg(proj)


def best_beta1(
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

    try:
        possible_paleo_slopes = np.array([
            paleo_slope(
                beta1_deg=beta1,
                beta2_deg=beta2_deg,
                sl2_deg=slope_deg
            )
            for beta1 in beta1_possible
        ])
        possible_tilts = np.array([
            great_circle_projection(beta2_deg, slope_deg) - great_circle_projection(*paleo_pair) for paleo_pair in zip(beta1_possible, possible_paleo_slopes)
        ])

        best_beta1 = beta1_possible[np.nanargmin(np.abs(possible_tilts))]
    except RuntimeWarning:  # raised if all NaNs
        return np.nan
    except ValueError:
        return np.nan

    return best_beta1


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

    try:
        possible_paleo_slopes = np.array([
            paleo_slope(
                beta1_deg=beta1,
                beta2_deg=beta2_deg,
                sl2_deg=slope_deg
            )
            for beta1 in beta1_possible
        ])
        possible_tilts = np.array([
            great_circle_projection(beta2_deg, slope_deg) - great_circle_projection(*paleo_pair) for paleo_pair in zip(beta1_possible, possible_paleo_slopes)
        ])

        best_tilt = possible_tilts[np.nanargmin(np.abs(possible_tilts))]
    except RuntimeWarning:  # raised if all NaNs
        return np.nan
    except ValueError:
        return np.nan

    return best_tilt


def tilt_check(
    test_az1: float,
    test_az2: float,
    test_sl2: float,
    paleo_azimuth_uncertainty: float = 7,
    minimum_offset: float = 7,
    max_abs_tilt: float = 5,
    plot_resolution: int = 3,
    plot_size: float = 5,
    plot_radius: float = 0.1,
    legend_position: tuple = (0.5, 0.5),
    full_tilt: bool = True,
    example_bearing=None,
    radial_ticks: bool = True
):  # type: ignore

    is_discordant = angular_difference(
        test_az1, test_az2) > paleo_azimuth_uncertainty

    bearing_range = np.arange(360 * plot_resolution) / plot_resolution

    test_rim = 0.9 * plot_radius * np.ones(len(bearing_range))

    test_tilt = []

    for bearing in bearing_range:
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
        subplot_kw={'projection': 'polar'}, figsize=(plot_size, plot_size), dpi=800)

    ax.set_theta_direction(-1)  # type: ignore
    ax.set_theta_offset(np.pi/2.0)  # type: ignore

    # radial limit
    plt.ylim(0, plot_radius)

    # plot az1 uncertainty range
    if minimum_offset != 0:
        ax.vlines(
            x=np.radians((
                test_az1 + minimum_offset,
                test_az1 - minimum_offset
            )),
            ymin=0,
            ymax=plot_radius,
            label=f"$\\theta_1 \pm$" + f"{minimum_offset}",
            color='red'
        )
        ax.vlines(
            x=np.radians((
                test_az1 + 180 - minimum_offset,
                test_az1 + 180 + minimum_offset
            )),
            ymin=0,
            ymax=plot_radius,
            color='red'
        )

    # plot measured az1
    ax.vlines(x=np.radians(test_az1), ymin=0, ymax=plot_radius,
              colors=['black'], label=f"$\\theta_1=$" + f"{test_az1}")

    # plot modern topographic attitude
    sns.scatterplot(
        x=[np.radians(test_az2)],
        y=[np.sin(np.radians(test_sl2))],
        label='$\\theta_2=$'
        + f'{test_az2}, \n'
        + '$\\varphi_2=$'
        + f'{test_sl2}',
        color='black',
        s=100,
        zorder=4)

    # nice radial ticks
    if radial_ticks:
        ax.set_yticks(np.linspace(0, plot_radius, 3)[1:])
    else:
        ax.set_yticks([])

    if is_discordant:
        palette = 'RdYlGn'
    else:
        palette = 'Blues'

    # plot calculated tilt by color around the rim. bearing + 180 to show direction TOWARD the center
    if full_tilt:
        for i, r in enumerate(np.linspace(0.9, 1.2, 60 * plot_resolution)):
            if i == 0:
                sns.scatterplot(
                    x=np.radians(bearing_range + 180),
                    y=test_rim,
                    hue=nptest_tilt,
                    linewidth=0,
                    palette=palette,
                    size=1,
                    # legend=False
                )
            else:
                sns.scatterplot(
                    x=np.radians(bearing_range + 180),
                    y=r * test_rim,
                    hue=nptest_tilt,
                    linewidth=0,
                    palette=palette,
                    size=1,
                    legend=False  # type: ignore
                )

    if example_bearing is not None:
        plt.vlines(
            x=np.radians(example_bearing),
            ymin=0,
            ymax=1,
            label='To Center',
            colors=['grey']
        )
        plt.vlines(
            x=np.radians((
                example_bearing + 90,
                example_bearing - 90,
            )),
            ymin=0,
            ymax=1,
            linestyles='dashed',
            colors=['grey', 'grey'],
            label='Tilt Axis'
        )

        example_beta2 = signed_angular_difference(
            ang2_deg=test_az2, ang1_deg=example_bearing + 180
        )
        example_beta1 = signed_angular_difference(
            ang2_deg=test_az1, ang1_deg=example_bearing + 180
        )

        example_beta1_boundary = best_beta1(
            beta1_deg=example_beta1,
            beta2_deg=example_beta2,
            slope_deg=test_sl2,
            paleo_azimuth_uncertainty=paleo_azimuth_uncertainty
        )

        example_tilt = minimum_tilt(
            beta1_deg=example_beta1,
            beta2_deg=example_beta2,
            slope_deg=test_sl2,
            paleo_azimuth_uncertainty=paleo_azimuth_uncertainty
        )

        example_paleo_slope = paleo_slope(
            beta1_deg=example_beta1_boundary,
            beta2_deg=example_beta2,
            sl2_deg=test_sl2
        )

        example_az1_boundary = example_bearing + 180 + example_beta1_boundary

        tilt_path_az = np.radians([
            example_az1_boundary, test_az2
        ])

        tilt_path_sl = np.sin(np.radians([
            example_paleo_slope, test_sl2
        ]))

        if not np.isnan(example_tilt):
            sns.lineplot(
                x=tilt_path_az,
                y=tilt_path_sl,
                label=f'Tilt: {round(example_tilt, 1)}'
            )

            sns.scatterplot(
                x=[np.radians(example_az1_boundary)],
                y=[np.sin(np.radians(example_paleo_slope))],
                label='$\\theta_1=$'
                + f'{test_az1}, \n'
                + '$\\varphi_1=$'
                + f'{round(example_paleo_slope, 1)}',
                color='black',
                s=100,
                marker='X',
                zorder=5)

    handles, labels = ax.get_legend_handles_labels()

    # label az1 center before uncertainty
    handles[0], handles[1] = handles[1], handles[0]
    labels[0], labels[1] = labels[1], labels[0]

    # this is a hack to remove the "1" that appears randomly in the legend in the for loop implementation of thick rim

    if full_tilt:
        if is_discordant and (example_bearing != None):
            del (handles[7])
            del (labels[7])
        else:
            del (handles[-1])
            del (labels[-1])

    ax.legend(
        handles,
        labels,
        loc='center',
        framealpha=0.8,
        bbox_to_anchor=legend_position
    )

    ax.set_axisbelow(True)


def plot_tilt_distance_dataset(df: pd.DataFrame, color=None, size=None, name: str = ''):

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
        size=size,
        color=color,
        label=name,
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
        self.calculate()

    def get_data_subset(self, sIDs: list) -> pd.DataFrame:
        return self.data.loc[sIDs]

    def calculate(self):

        self.data['dist_m'] = self.data.apply(
            lambda row: great_circle_distance(
                lat1_deg=row['LAT'], lon1_deg=row['LON'],
                lat2_deg=self.lat, lon2_deg=self.lon
            ),
            axis=1
        )

        self.data['dist_km'] = self.data.apply(
            lambda row: row['dist_m'] / 1000,
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

        self.data['alignment'] = self.data.apply(
            lambda row: np.cos(np.radians(row['beta1'])),
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

    def plot_tilt(
        self,
        pops: list[Population],
        max_alignment: float = 1,
        min_alignment: float = -1,
        size=None
    ):

        colors = itertools.cycle(sns.color_palette())  # type: ignore

        for pop in pops:
            next_color = next(colors)

            subset = self.data.loc[pop.sIDs].copy()

            above_max = subset['alignment'] >= max_alignment
            below_min = subset['alignment'] <= min_alignment

            subset.loc[(above_max & below_min), 'tilt'] = np.nan

            num_tiltable_offset = len(subset[subset['tilt'].notnull()])

            plot_tilt_distance_dataset(
                subset,
                next_color,
                name=f"Pop: {pop.name}. $f = ${num_tiltable_offset}/{len(subset)}",
                size=size
            )


def make_center(cID, centers, samples, paleo_azimuth_uncertainty):
    return Center(
        cID=cID,
        lat=centers.loc[cID, 'LAT'],
        lon=centers.loc[cID, 'LON'],
        data=samples.copy(),
        paleo_azimuth_uncertainty=paleo_azimuth_uncertainty
    )


def summit_score(df: pd.DataFrame) -> dict:
    try:
        score = {'summit_score': mean(np.abs(df['beta1']))}
    except:
        score = {'summit_score': np.nan}
    return score


def fit_mogi_function(df: pd.DataFrame, full_output: bool = False) -> tuple:

    # initial guess
    if np.mean(df['tilt']) < 0:
        p0 = -TEST_EPV_J, TEST_DEPTH_M
    else:
        p0 = TEST_EPV_J, TEST_DEPTH_M

    fit = curve_fit(
        f=mogi_tilt,
        xdata=df['dist_m'],
        ydata=df['tilt'],
        p0=p0,
        maxfev=MAX_ITERATIONS,
        method='lm',
        full_output=full_output
    )

    return fit

# criteria functions to return subset_sizes and analytical regression results. These need to be passed as functions df -> dict of scores. But since both criteria depend on a alignment threshold, each df -> dict function is wrapped in a float -> Callabe function


def prepend_dict_keys(dict, prefix):
    return {prefix + str(key): val for key, val in dict.items()}


def subset_sizes(alignment_threshold_degrees: float = 0) -> Callable[[pd.DataFrame], dict]:
    def func(df: pd.DataFrame) -> dict:

        alignment_threshold = np.cos(np.radians(alignment_threshold_degrees))
        offset = df[np.abs(df['alignment']) < alignment_threshold]

        tiltable = df[df['tilt'].notnull()]
        tiltable_offset = offset[offset['tilt'].notnull()]

        scores = {
            f'tiltable': len(tiltable) / len(df),
            f'offset': len(offset) / len(df),
            f'tiltable_offset': len(tiltable_offset) / len(df),
        }

        scores = prepend_dict_keys(scores, f'{alignment_threshold_degrees}_')

        return scores
    return func


def mogi_maximum(
        epv_J: float,
        depth_m: float,
        dist_m_interval: tuple[float, float]) -> dict[str, float]:

    dist_m_array = np.linspace(*dist_m_interval, 1000)
    tilt_array = [mogi_tilt(dist, epv_J, depth_m) for dist in dist_m_array]
    max_index = np.argmax(np.abs(tilt_array))
    max_tilt_dist_m = dist_m_array[max_index]
    max_tilt = tilt_array[max_index]

    return {'tilt': max_tilt, 'dist_m': max_tilt_dist_m}


def inflation_score(alignment_threshold_degrees: float = 0) -> Callable[[pd.DataFrame], dict]:
    def func(df: pd.DataFrame) -> dict:
        alignment_threshold = np.cos(np.radians(alignment_threshold_degrees))
        offset = df[np.abs(df['alignment']) < alignment_threshold]
        tiltable = offset[offset['tilt'].notnull()]
        dist_m_interval = (min(tiltable['dist_m']), max(tiltable['dist_m']))

        scores = {
            'inflation': np.nan,
            'depth_m': np.nan,
            'log10_epv': np.nan,
            'max_tilt': np.nan,
            'rmse': np.nan,
        }

        try:
            params, _, infodict, _, _ = fit_mogi_function(
                df=tiltable,
                full_output=True
            )
            epv_J, depth_m = params
            rmse = np.sqrt(mean_squared_error(
                y_pred=infodict['fvec'],
                y_true=tiltable['tilt']
            ))

            max_tilt = mogi_maximum(epv_J, depth_m, dist_m_interval)

            # rewrite scores in dict
            scores['inflation'] = epv_J > 0
            scores['log10_epv'] = np.log10(np.abs(epv_J))
            scores['depth_m'] = depth_m
            scores['max_tilt'] = max_tilt['tilt']
            scores['rmse'] = rmse  # type: ignore

        # catch regression failure
        except OptimizeWarning:  # does not converge
            pass
        except RuntimeError:
            pass
        except ValueError:
            pass
        except TypeError:
            pass

        scores = prepend_dict_keys(scores, f'{alignment_threshold_degrees}_')
        return scores
    return func


@dataclass
class Criterion:
    func: Callable
    pop: Population


def evaluate_center(center: Center, crit: Criterion):
    pop_subset = center.data.loc[crit.pop.sIDs]
    return crit.func(pop_subset)

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
        self.dist_m = self.pos2[0]  # type: ignore
        self.dist_km = self.dist_m / 1000

        # initial and displaced slopes (positive downward from center)
        # index [1] is z component; [0] is r component
        self.slope1 = deg(arctan2(-self.shape1[1], self.shape1[0]))
        self.slope2 = deg(arctan2(-self.shape2[1], self.shape2[0]))

        self.tilt = self.slope2 - self.slope1


model_path = "../GEOL192-Model/data/"


# paleo-edifice spline data
topo = np.genfromtxt(f'{model_path}z1.csv', delimiter=",").T


def model_pos1_from_csv(name: str, topo: np.ndarray) -> np.ndarray:
    r = np.genfromtxt(model_path + "rdisp_" + name, delimiter=",")[:, 0]
    z = np.interp(r, *topo, right=0)  # interpolate z1 into topography
    return np.array([r, z]).T


def model_disp_from_csv(name: str) -> np.ndarray:
    r = np.genfromtxt(f'{model_path}rdisp_{name}', delimiter=",")[:, 1]
    z = np.genfromtxt(f'{model_path}zdisp_{name}', delimiter=",")[:, 1]
    return np.array([r, z]).T


def read_model_data(params: dict, topo: np.ndarray) -> dict[str, np.ndarray]:

    filename = f"depth_{params['depth']}_radius_{params['radius']}_aspect_{params['aspect']}_pmult_{params['pmult']}_grav_{int(params['grav'])}_topo_{int(params['topo'])}.csv"

    pos1 = model_pos1_from_csv(filename, topo)
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


def plot_ellipsoidal_reservoir(
        depth_km: float,
        max_elev_km: float,
        radius_km: float,
        aspect: float,
        label: str,
) -> None:

    center_z_km = max_elev_km - depth_km
    half_height_km = radius_km * aspect

    t = np.linspace(-np.pi/2, np.pi/2, 100)
    plt.plot(
        radius_km * np.cos(t),
        center_z_km + half_height_km * np.sin(t),
        label=label
    )


@dataclass
class NumericalModel:
    params: dict
    pos1: np.array = field(repr=False)  # type: ignore
    disp: np.array = field(repr=False)  # type: ignore
    rmse: float = np.nan

    def __post_init__(self) -> None:

        self.set_params()
        self.nodes = [Node(*node) for node in zip(self.pos1, self.disp)]
        self.consecutive_node_pairs = zip(self.nodes[:-1], self.nodes[1:])
        self.edges = [Edge(*pair) for pair in self.consecutive_node_pairs]
        self.data = pd.DataFrame([vars(edge) for edge in self.edges])

    def set_params(self) -> None:
        self.radius = self.params['radius']
        self.half_height = self.radius * self.params['aspect']

        self.over_pressure = self.params['pmult'] * \
            self.params['depth'] * ROCK_DENSITY_KG_M3 * MARS_GRAVITY_M_S2

        self.res_vol = (4 / 3) * np.pi * self.radius**2 * self.half_height

        self.epv = self.over_pressure * self.res_vol
        self.depth = self.params['depth'] + self.half_height

    def plot_numerical_tilt(self, label: str = None) -> None:  # type: ignore
        sns.lineplot(data=self.data, x='dist_km', y='tilt', label=label)

    def plot_numerical_displacement(
            self,
            r_label: str = None,  # type: ignore
            z_label: str = None  # type: ignore
    ) -> None:
        sns.lineplot(data=self.data, x='dist_km', y='disp_r', label=r_label)
        sns.lineplot(data=self.data, x='dist_km', y='disp_z', label=z_label)

    def plot_section(self, max_elev_km: float, label: str) -> None:
        plot_ellipsoidal_reservoir(
            depth_km=self.params['depth'] / 1000,
            max_elev_km=max_elev_km,
            radius_km=self.params['radius'] / 1000,
            aspect=self.params['aspect'],
            label=label
        )


def unpack_param_combinations(dict_of_lists: dict[str, list]) -> list[dict[str, float]]:
    keys = dict_of_lists.keys()
    all_vals = list(itertools.product(*dict_of_lists.values()))
    list_of_dicts = [dict(zip(keys, vals)) for vals in all_vals]
    return list_of_dicts


def make_numerical_model(params: dict[str, list], topo: np.ndarray) -> NumericalModel:
    model = NumericalModel(
        params=params,
        pos1=read_model_data(params, topo)['pos1'],
        disp=read_model_data(params, topo)['disp']
    )
    return model


def numerical_rmse(
    model: NumericalModel,
    map_tilt_df: pd.DataFrame
) -> float:

    possible_subset = map_tilt_df[map_tilt_df['tilt'].notnull()]

    tilt_map = possible_subset['tilt'].tolist()
    dist_km_map = possible_subset['dist_km'].tolist()

    tilt_numerical_model = model.data['tilt'].tolist()
    dist_km_numerical_model = model.data['dist_km'].tolist()

    tilt_predicted = np.interp(
        x=dist_km_map,  # type: ignore
        xp=dist_km_numerical_model,
        fp=tilt_numerical_model
    )

    rmse = np.sqrt(mean_squared_error(
        y_true=tilt_map,
        y_pred=tilt_predicted
    ))

    return rmse  # type: ignore


@dataclass
class ParamSweep:
    params: list[dict]
    topo: np.ndarray

    def __post_init__(self) -> None:
        self.models = []
        for model_params in self.params:
            try:
                self.models.append(make_numerical_model(model_params, topo))
            except:
                pass

    def sort_models_by_rmse(self, center: Center, sIDs: list) -> None:
        data = center.data.loc[sIDs]
        data_no_nulls = data[data['tilt'].notnull()]
        for model in self.models:
            try:
                model.rmse = numerical_rmse(
                    model, data_no_nulls  # type: ignore
                )
            except:
                pass

        self.models.sort(key=lambda model: model.rmse)


def plot_mogi_tilt(
    epv_J: float,
    depth_m: float,
    dist_m_interval: tuple[float, float] = (0, 100_000),
    label: str = '',
    color: str = 'black'
) -> None:
    dist_m_array = np.linspace(*dist_m_interval, 1000)
    tilt = mogi_tilt(dist_m_array, epv_J, depth_m)
    dist_km = dist_m_array / 1000

    sns.lineplot(x=dist_km, y=tilt, c=color, label=label)
