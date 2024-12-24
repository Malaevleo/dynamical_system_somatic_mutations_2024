import matplotlib.pyplot as plt
import matplotlib.style
import scipy.integrate._ivp.ivp
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import json
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


def print_methods():
    print('RK45, RK23, Radau, DOP853, LSODA, BDF')


def print_styles():
    print(matplotlib.style.available)


class SomaticLS(object):
    def __init__(
            self,
            organ: str = 'liver',
            method: str = 'Radau',
            start_time: float = 0,
            end_time: float = 300,
            include_mutants: bool = False,
            equation: str = 'one',
            custom_conf=None,
            style: str = 'bmh'):
        """
            Class for lifespan simulation based on the only hallmark of aging - somatic mutations.

            ######
            Args: organ for simulation, method for solve_ivp, 
                  start and end times in years, include alive mutants in Model 1 system or not,
                  type of system, whether to print config, custom_conf - specify your custom parameters for a system.

            ######
            Output: constructor
        """

        set_eqs = ['one', 'two', 'combined'] 

        self.style = style
        plt.style.use(self.style)

        organs = ['liver', 'mouse liver', 'lungs', 'spinal cord', 'cirrhosis']
        self.method = method
        self.organ = organ
        self.custom = False

        if self.organ not in organs:
            print('This organ is not specified. Use your custom config to solve the system')
            if custom_conf is None:
                raise ValueError('Please specify your custom config when using not built-in organs.')
            else:
                self.custom_conf, self.custom_init, self.custom_thr = self.custom_params(custom_conf)
                self.custom = True

        self.conf_ = self.set_config()

        if not include_mutants:
            self.conf_['theta'] = 0
            self.conf_['z'] = 0

        self.end_time = end_time
        self.coeff = 21.5 / (365.25 * 24)
        self.start = int(start_time / self.coeff)
        self.end = int(np.round(end_time / self.coeff))
        self.span = -self.start + self.end
        self.t = np.linspace(self.start, self.end, self.end - self.start)

        self.include = include_mutants
        self.equation = equation

        self.init = self.get_initial_conditions()
        self.threshold = self.get_threshold()
        self.model = self.get_model()
        self.life = self.lifespan(verbose=False) if isinstance(self.lifespan(verbose=False), np.int64) else None

        if self.equation not in set_eqs:
            raise ValueError('Please use only types (one, two) for equation settings')
        
        if (self.organ == 'spinal cord') & (self.equation == 'two'):
            print('======= No two-equation system for spinal cord. Changing to Model 1 equation. =======')
            self.equation = 'one'

    def __eq__(self, other):
        if (self.life is not None) & (other.life is not None):
            return self.life == other.life
        else:
            if self.life is None:
                bad = self.organ
            else:
                bad = other.organ
            print('One of the organs or both of them haven\'t died. Try to set higher simulation time for {}'.format(bad))

    def __lt__(self, other):
        if (self.life is not None) & (other.life is not None):
            return self.life < other.life
        else:
            if self.life is None:
                bad = self.organ
            else:
                bad = other.organ
            print('One of the organs or both of them haven\'t died. Try to set higher simulation time for {}'.format(bad))
    
    def __le__(self, other):
        if (self.life is not None) & (other.life is not None):
            return self.life <= other.life
        else:
            if self.life is None:
                bad = self.organ
            else:
                bad = other.organ
            print('One of the organs or both of them haven\'t died. Try to set higher simulation time for {}'.format(bad))

    def __gt__(self, other):
        if (self.life is not None) & (other.life is not None):
            return self.life > other.life
        else:
            if self.life is None:
                bad = self.organ
            else:
                bad = other.organ
            print('One of the organs or both of them haven\'t died. Try to set higher simulation time for {}'.format(bad))

    def __ge__(self, other):
        if (self.life is not None) & (other.life is not None):
            return self.life >= other.life
        else:
            if self.life is None:
                bad = self.organ
            else:
                bad = other.organ
            print('One of the organs or both of them haven\'t died. Try to set higher simulation time for {}'.format(bad))

    def __str__(self):
        return self._print_config()

    def __getitem__(self, item):
        if item == 'parameters':
            return self.conf_ if not self.custom else self.custom_conf
        elif item == 'lifespan':
            return {'Mitosis': self.life, 'Years': int(self.life * self.coeff)} if self.life is not None else None
        elif item == 'population':
            return self.calculate_population()
        elif item == 'initial':
            return self.init if not self.custom else self.custom_init
        elif item == 'threshold':
            return self.threshold if not self.custom else self.custom_thr
        elif item == 'is dead':
            return isinstance(self.life, np.int64)
        elif item == 'start time':
            return self.start
        elif item == 'end time':
            return self.end
        else:
            raise NameError('Cannot access this attribute')

    def _print_config(self) -> str:
        return f'''
              CONFIG FOR SIMULATION HAS BEEN CREATED
              {'-'*40}
              Final parameters set:
              --organ: {self.organ},
              --start: {np.round(self.start*(21.5 / (365.25 * 24)))} years,
              --end: {np.round(self.end*(21.5 / (365.25 * 24)))} years,
              --type of system: Model {self.equation} equation system,
              --solver method: {self.method},
              --include mutants: {self.include}
              {'-'*40}
              '''
    
    @staticmethod
    def _clear_dict(val: dict) -> Dict:
        l_cols = ['parameters', 'initial conditions', 'threshold']
        list_essential = ['K', 'M', 'alpha', 'sigma', 'beta', 'eps', 'r', 'g', 'z', 'theta']
        list_total = []
        for el in l_cols:
            if isinstance(val[el], dict):
                list_total.append([i for i in val[el].keys() if np.isnan(val[el][i]) == 1])
            elif np.isnan(val[el]) == 1:
                raise ValueError('There shouldn\'t be NaNs')
        if len(list_total[-1]) != 0: 
            for el in list_total[2]:
                val['threshold'].pop(el)
            for el in list_total[-2]:
                val['initial conditions'].pop(el)
        else:
            for el in list_total[-1]:
                val['initial conditions'].pop(el)
        if len(list_total[0]) != 0:
            for el in list_total[0]:
                val['parameters'].pop(el)
        else:
            for el in list_essential:
                if np.isnan(val['parameters'][el]) == 1:
                    raise ValueError('There shouldn\'t be NaNs')
        return val
    
    def custom_params(self, uploaded=None) -> Tuple[dict, dict, float]:
        """
        Upload your own parameters, threshold and initial conditions of a model

        ######
        Args: uploaded - either a path to a .csv, .xlsx or .json file or a python dictionary, pandas dataframe

        ######
        Output: your parameters, initial conditions and threshold
        """

        if isinstance(uploaded, str):
            if uploaded[-3:] == 'csv':
                custom_conf = pd.read_csv(uploaded).set_index('Unnamed: 0')
            elif uploaded[-4:] == 'xlsx':
                custom_conf = pd.read_excel(uploaded).set_index('Unnamed: 0')
            elif uploaded[-4:] == 'json':
                with open(uploaded) as json_file:
                    custom_conf = json.load(json_file)
                custom_conf = self._clear_dict(custom_conf)
                params = custom_conf['parameters']
                init = custom_conf['initial conditions']
                thr = custom_conf['threshold']
                return params, init, thr
            else:
                raise NameError('File should be either excel, csv of json')
            custom_conf = custom_conf.to_dict()
            custom_conf = self._clear_dict(custom_conf)
        elif isinstance(uploaded, dict):
            custom_conf = self._clear_dict(uploaded)
            params = custom_conf['parameters']
            init = custom_conf['initial conditions']
            thr = custom_conf['threshold']
            return params, init, thr
        elif isinstance(uploaded, pd.core.frame.DataFrame):
            custom_conf = uploaded.to_dict()
            custom_conf = self._clear_dict(custom_conf)
        else:
            raise TypeError('Argument should either be a path to csv, excel, json file or a dict, pd.DataFrame')

        params = custom_conf['parameters']
        init = custom_conf['initial conditions']
        thr = custom_conf['threshold']['K']
        return params, init, thr
    
    def set_config(self) -> Dict:
        configs = {
            'liver': {
                'sigma': 0.117, 'K': 2e11, 'M': 2e11/94000, 'r': 4/407, 'eps': 0.064,
                'alpha': 3.5e-9*9.6e-3, 'beta': 1.83e-9*9.6e-3, 'g': 4/407, 'z': 0.9, 'theta': 0.239
                },
            'mouse liver': {
                'sigma': 0.117, 'K': 3.37e8, 'M': 3.37e8/94000, 'r': 63/407, 'eps': 0.064,
                'alpha': 35*3.5e-9, 'beta': 35*1.83e-9, 'g': 63/407, 'z': 0.9, 'theta': 0.239
                },
            'lungs': {
                'sigma': 0.117, 'K': 10.5e9, 'M': 0.07*10.5e9, 'r': 0.001/407, 'eps': 0.007,
                'alpha': 6.392476819356688e-12, 'beta': 6.392476819356688e-12 / 1.9126, 'g': 0.001/407,
                'z': 0.9, 'theta': 0.239
                },
            'spinal cord': {
                'sigma': 0.117, 'K': 222e6, 'M': 0, 'r': 0, 'eps': 0,
                'alpha': 0.9047619*3.5e-9*0.0013563, 'beta': 0, 'g': 0, 'z': 0.9, 'theta': 0.239
                },
            'cirrhosis': {
                'sigma': 0.117, 'K': 2e11, 'M': 2e11/94000, 'r': 4/407, 'eps': 0.064,
                'alpha': 1.5*3.5e-9*9.6e-3, 'beta': 1.5*1.83e-9*9.6e-3, 'g': 4/407, 'z': 0.9, 'theta': 0.239
                }
        }
        return configs.get(self.organ, configs['liver'])
    
    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1 / (1 + np.exp(-10 * (z - 0.5)))

    def restart(self, memory: bool = False, *args, **kwargs):
        """
        Restart the initialisation of a system without unnecessary verbosity

        ######
        Args: any of the constructor arguments,
              memory - whether to change given attributes without restarting the constructor

        ######
        Output: new exemplar
        """

        if not memory:
            return SomaticLS(*args, **kwargs)
        else:
            args_passed = tuple(kwargs.keys())

            for el in args_passed:
                if el == 'start_time':
                    self.start = kwargs[el] / self.coeff
                elif el == 'end_time':
                    self.end = kwargs[el] / self.coeff
                elif el == 'method':
                    self.method = kwargs[el]
                elif el == 'include_mutants':
                    self.include = kwargs[el]
                elif el == 'organ':
                    self.organ = kwargs[el]
                elif el == 'equation':
                    self.equation = kwargs[el]
                elif el == 'custom_conf':
                    self.custom_conf, self.custom_init, self.custom_thr = self.custom_params(kwargs[el])
                elif el == 'style':
                    self.style = kwargs[el]
                else:
                    raise ValueError('No such attribute')

    @staticmethod
    def _model_one(t, y, s, K, M, r, e, a, b, g, z, d) -> List:
        X, C, F, m = y
        m1 = 0.5 * s * (1 - X / K)
        dXdt = r * X * (1 - X / K) - a * X - m1 * (r * C * (1 - C / K) + a * X) ** 2 * t ** 2
        dCdt = r * C * (1 - C / K) + z * a * X - d * C
        dFdt = (1 - z) * a * X + d * C
        dmdt = m1 * (r * C * (1 - C / K) + a * X) ** 2 * t ** 2
        return [dXdt, dCdt, dFdt, dmdt]

    def _model_two(self, t, y, s, K, M, r, e, a, b, g, z, d) -> List:
        X, Y, m = y
        m1 = 0.5 * s * (1 - (X + Y) / (K + M))
        dXdt = (r * X + 2 * e * Y * self._sigmoid(Y / M)) * (1 - X/K) - a * X - m1 * (a * X + b * Y) ** 2 * t ** 2
        dYdt = g * Y * (1 - Y / M) - e * Y * self._sigmoid(Y / M) - b * Y
        dmdt = m1 * (a * X + b * Y) ** 2 * t ** 2
        return [dXdt, dYdt, dmdt]
    
    def _model_comb(self, t, y, s, K, M, r, e, a, b, g, z, d) -> List:
        X, Y, C, F, T, G, m = y
        m1 = 0.5 * s * (1 - (X + Y) / (K + M))
        dXdt = (r * X + 2 * e * Y*self._sigmoid(Y/M))* (1 - X / K) - a * X - m1 * (a * X + b * Y + r * C * (1 - C / K) + r * T * (1 - T / M)) ** 2 * t ** 2
        dYdt = g * Y * (1 - Y / M) - e * Y * self._sigmoid(Y/M) - b * Y
        dCdt = (r * C + 2 * e * T) * (1 - C / K) + z * a * X - d * C
        dFdt = (1 - z) * a * X + d * C
        dTdt = r * T * (1 - T / M) + z * b * Y - d * T - e * T
        dGdt = (1 - z) * b * Y + d * T
        dmdt = m1 * (a * X + b * Y + r * C * (1 - C / K) + r * T * (1 - T / M)) ** 2 * t ** 2
        return [dXdt, dYdt, dCdt, dFdt, dTdt, dGdt, dmdt]
    
    def get_initial_conditions(self) -> Dict:
        K = self.conf_['K']
        M = self.conf_['M']

        if self.equation == 'one':
            initial = {
                'liver': {
                    'K': K, 'C': 0, 'F': 0, 'm': 0
                    },
                'lungs': {
                    'K': K, 'C': 0, 'F': 0, 'm': 0
                    },
                'spinal cord': {
                    'K': K, 'C': 0, 'F': 0, 'm': 0
                    },
                'mouse liver': {
                    'K': K, 'C': 0, 'F': 0, 'm': 0
                    },
                'cirrhosis': {
                    'K': K, 'C': 0, 'F': 0, 'm': 0
                    }
            }
        elif self.equation == 'two':
            initial = {
                'liver': {
                    'K': K, 'M': M, 'm': 0
                    },
                'lungs': {
                    'K': K, 'M': M, 'm': 0
                    },
                'mouse liver': {
                    'K': K, 'M': M, 'm': 0
                    },
                'cirrhosis': {
                    'K': K, 'M': 0, 'm': 0
                    }
            }
        
        elif self.equation == 'combined':
            initial = {
                'liver': {
                    'K': K, 'M': M, 'C': 0, 'F': 0, 'T':0, 'G':0, 'm': 0
                    },
                'lungs': {
                    'K': K, 'M': M, 'C': 0, 'F': 0, 'T':0, 'G':0, 'm': 0
                    },
                'spinal cord': {
                    'K': K, 'M': M, 'C': 0, 'F': 0, 'T':0, 'G':0, 'm': 0
                    },
                'mouse liver': {
                    'K': K, 'M': M, 'C': 0, 'F': 0, 'T':0, 'G':0, 'm': 0
                    },
                'cirrhosis': {
                    'K': K, 'M': M, 'C': 0, 'F': 0, 'T':0, 'G':0, 'm': 0
                    }
            }

        return initial.get(self.organ, initial['liver'])
    
    def get_model(self):
        if self.equation == 'one':
            return self._model_one
        elif self.equation == 'two':
            return self._model_two
        elif self.equation == 'combined':
            return self._model_comb
    
    def calculate_population(self) -> scipy.integrate._ivp.ivp.OdeResult:
        """
            Calculate populations by solving differential equation system numerically.

            ######
            Args: instance attributes

            ######
            Output: solve_ivp object with calculated cell populations
        """

        if not self.custom:
            conf = tuple(self.conf_.values())
            initial = self.init
        else:
            conf = tuple(self.custom_conf.values())
            initial = self.custom_init

        solution = solve_ivp(
            self.model,
            t_span=(self.start, self.end), 
            y0=list(initial.values()), 
            t_eval=self.t, 
            method=self.method,
            args=conf, 
            dense_output=False, 
            atol=1e-6,
            rtol=1e-6)
        
        return solution  
    
    def get_threshold(self) -> float:
        thr = {
            'liver': 0.3,
            'mouse liver': 0.3,
            'lungs': 0.4,
            'spinal cord': 0.9
        }
        return thr.get(self.organ, thr['liver'])

    def lifespan(
            self,
            custom_solution: scipy.integrate._ivp.ivp.OdeResult = None,
            verbose: bool = True) -> int:
        """
            Calculate a lifespan of somatic population given a cutoff value.

            ######
            Args: custom_solution - your own solve_ivp object,
                  verbose - whether to print lifespans and regen times

            ######
            Output: a moment when the population reaches a cutoff value.
        """

        if self.custom:
            thr = self.custom_thr
            K = self.custom_conf['K']
        else:
            thr = self.threshold
            K = self.conf_['K']

        if custom_solution is None:
            arr = self.calculate_population()
        else:
            arr = custom_solution

        arr_ = arr.y[0]
        l = np.where(arr_ <= thr*K)[0]
        k = np.where(arr_ > 0.99*K)[0]

        if len(l) != 0:
            if verbose:
                if self.organ == 'mouse liver':
                    print('Life expectancy (years):', np.round(l[0] * self.coeff, 2))
                else:
                    print('Life expectancy (years):', np.round(l[0] * self.coeff))
                print('-'*50)
            ls = l[0]
        else:
            if verbose:
                print('Haven\'t died')
                print('-'*50)
            ls = None

        if verbose:
            if len(k) != 0:
                if arr.y[0][0] == K:
                    print('No resection')
                else:
                    print('Time of regeneration of somatic cells (years):', np.round(k[0] * self.coeff, 2))
            else:
                if arr.y[0][0] != K:
                    print('Haven\'t regenerated')
        
        return ls
    
    @property
    def calc_entropy_prod(self):
        if (self.equation != 'one') and (self.include != False): 
            raise ValueError('Only for Model 1 without dividing mutants.')
        else:
            X = self.calculate_population().y[0]
            X_dot = np.diff(X)
            sigma, K, _, r, _, _, alpha, _, _, _ = self.set_config().values()
            first = X_dot*(r*(1 - X[:-1]/K) - alpha*X[:-1] - sigma*X[:-1]*(alpha*self.t[:-1])**2 + (1.5/K)*sigma*(alpha*X[:-1]*self.t[:-1])**2)
            second = alpha*X[:-1]*(sigma*alpha*X[:-1]*self.t[:-1]*(1 - X[:-1]/K))
            return X_dot*(first - second)


    
    def plot_curves(
            self,
            population: str = 'Somatic',
            view_all: bool = False,
            proportions: bool = True,
            plot_thr: bool = True,
            root: int = 1,
            derivative: bool = False,
            logder: bool = False,
            lim: List|Tuple = (0, 300)) -> None:
        """
            Plot results of a simulation.

            ######
            Args: population - type of population, view_all - show only till the moment of death or not, 
                  proportions - plot as population/(population limit), plot_thr - plot cutoff value,
                  logder - semilogy for mortality function derivative.

            ######
            Output: plots, lifespan
        """

        if not self.custom:
            thr = self.threshold
            K = self.conf_['K']
        else:
            thr = self.custom_thr
            K = self.custom_conf['K']
        life = self.lifespan()
        arr = self.calculate_population()

        if self.equation == 'one':
            if population == 'Somatic':
                Y = arr.y[0]
            elif population == 'Cirrhosis comparison':
              if (self.organ != 'liver'):
                raise ValueError('Comparison is applicable only for the healthy human liver.')
              else: 
                if not self.custom:
                    original_conf = self.conf_.copy()
                else:
                    original_conf = self.custom_conf.copy()
                arr_default = self.calculate_population()
                Y = arr_default.y[0]
                if not self.custom:
                    self.conf_['alpha'] *= 1.5
                    self.conf_['beta']  *= 1.5
                else:
                    self.custom_conf['alpha'] *= 1.5
                    self.custom_conf['beta']  *= 1.5
                arr_cirr = self.calculate_population()
                Y_cirr = arr_cirr.y[0]
                if not self.custom:
                    self.conf_ = original_conf
                else:
                    self.custom_conf = original_conf

            elif population == 'Alive mutants':
                Y = arr.y[1]
                proportions = False
                plot_thr = False
            elif population == 'Dead mutants':
                Y = arr.y[2]
                proportions = False
                plot_thr = False
            elif population == 'Mortality function':
                self._plot_mortality(arr, life, derivative, logder)
                return None
            elif population == 'Mortality phase':
                self._plot_mortality_phase(arr, K, proportions, derivative, logder)
                return None
            elif population == 'Stable points':
                self._plot_stable_points(root, proportions, plot_thr)
                return None
            elif population == 'Entropy production':
                dS = self.calc_entropy_prod
                if logder == False:
                    D = -dS
                    title = r'$ \dot S(t)$'
                else:
                    D = np.log10(-dS)
                    title = r'$\log_{10} \dot S(t)$'
                plt.plot(self.t[:-1]*self.coeff, D, c = 'r')
                plt.title('Entropy production over time')
                plt.grid('True')
                plt.ylabel(title)
                plt.xlabel(r'$t$')
                plt.xlim(lim[0], lim[1])
                plt.show()
                return None

            else:
                raise NameError('There is no such population in this system')
            
        elif (self.equation == 'two') | (self.equation == 'combined'):
            M = self.conf_['M']
            if population == 'Somatic':
                Y = arr.y[0]
            elif population == 'Stem':
                Y = arr.y[1]
            elif population == 'Cirrhosis comparison':
              if (self.organ != 'liver'):
                raise ValueError('Comparison is applicable only for the healthy human liver.')
              else: 
                if not self.custom:
                    original_conf = self.conf_.copy()
                else:
                    original_conf = self.custom_conf.copy()
                arr_default = self.calculate_population()
                Y = arr_default.y[0]
                if not self.custom:
                    self.conf_['alpha'] *= 1.5
                    self.conf_['beta']  *= 1.5
                else:
                    self.custom_conf['alpha'] *= 1.5
                    self.custom_conf['beta']  *= 1.5
                arr_cirr = self.calculate_population()
                Y_cirr = arr_cirr.y[0]
                if not self.custom:
                    self.conf_ = original_conf
                else:
                    self.custom_conf = original_conf

            elif population == 'Mortality function':
                self._plot_mortality(arr, life, derivative, logder)
                return None
            elif population == 'Mortality phase':
                self._plot_mortality_phase(arr, K, proportions, derivative, logder)
                return None
            elif population == 'Stable points':
                raise NameError('There are no analytic solutions of this system -> there are no stable points')
            else:
                raise NameError('There is no such population in this system')
        
        plt.figure(figsize = (10,8))
        if proportions:
            if population == 'Somatic':
                plt.plot(self.t*self.coeff, Y/K, label=population)
            elif population == 'Stem':
                plt.plot(self.t*self.coeff, Y/M, label=population)
            elif population == 'Cirrhosis comparison':
                plt.plot(self.t*self.coeff, Y/K, label='Healthy liver')
                plt.plot(self.t * self.coeff, Y_cirr / K, label=population)
            plt.axhline(thr, ls='--', color='r', label='Threshold')
        else:
            plt.plot(self.t*self.coeff, Y, label=population)
            if plot_thr and (population == 'Somatic'):
                plt.axhline(thr*K, ls='--', color='r', label='Threshold')
            elif plot_thr and (population == 'Stem'):
                plt.axhline(thr*M, ls='--', color='r', label='Threshold')
            elif plot_thr and (population == 'Cirrhosis comparison'):
                plt.axhline(thr*K, ls='--', color='r', label='Threshold')

        plt.xlabel('Years')

        if proportions:
            plt.ylabel('Population/Capacity')
        else:
            plt.ylabel('Population')

        if not view_all:
            if isinstance(life, np.int64):
                plt.xlim(self.start, life*self.coeff)

        plt.grid(True)
        plt.legend()
        plt.show()
    
    def _plot_mortality(self, arr: scipy.integrate._ivp.ivp.OdeResult, life, derivative: bool, logder: bool) -> None:
        if self.equation == 'one':
            Y = arr.y[3]
        elif self.equation == 'two':
            Y = arr.y[2]
        else:
            Y = arr.y[-1]

        deriv = np.diff(Y)

        maximum = np.argmax(deriv)
        max_der = np.max(deriv)

        if not derivative:
            plt.plot(self.t[:-1]*self.coeff, deriv)
            plt.xlabel('Years')
            plt.ylabel('M(t)')
            plt.title('Mortality function')
            plt.axvline(maximum * self.coeff, color='r', ls='--')
            plt.axhline(max_der, ls='-.', color='g')
            plt.grid(True)
            plt.show()
        else:
            der = np.diff(deriv)
            der = savgol_filter(der, 550, 5)
            max_ = np.max(der)
            max_der_ = np.argmax(der)
            _, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)

            axes[0].plot(self.t[:-2]*self.coeff, deriv[:-1])

            if not logder:
                axes[1].plot(self.t[:-2]*self.coeff, der)
            else:
                axes[1].semilogy(self.t[:-2]*self.coeff, der)

            axes[1].set_xlabel(r'$t(years)$', fontsize = 20)
            axes[1].set_ylabel(r'$\frac{\partial \mu(X,t)}{ \partial t}$', fontsize = 24)

            axes[0].set_xlabel(r'$t(years)$', fontsize = 20)
            axes[0].set_ylabel(r'$\mu(X,t)$', fontsize = 20)

            axes[0].set_title('Mortality function', fontsize = 18)
            axes[1].set_title('Mortality function derivative', fontsize = 18)

            axes[0].axvline(maximum*self.coeff, color='r', ls='--')
            axes[1].axvline(max_der_*self.coeff, color='r', ls='--')
            axes[1].axhline(max_, ls='-.', color='g')
            axes[0].axhline(max_der, ls='-.', color='g')
            axes[0].grid(True)
            axes[1].grid(True)
            print('Max derivative value:', np.round(max_, 0))
            print('Max derivative moment:', np.round(max_der_*self.coeff, 0))

            if isinstance(life, int):
                print(
                    'Ratio of max derivative to total lifespan in %:',
                    np.round((max_der_ * self.coeff * 100)/(life * self.coeff), 1)
                )

        print('Max value:', np.round(max_der, 0))
        print('Max moment:', np.round(maximum * self.coeff, 0))

        if isinstance(life, int):
            print('Ratio of max to total lifespan in %:', np.round((maximum * self.coeff * 100)/(life * self.coeff), 1))

    def _plot_mortality_phase(
            self,
            arr: scipy.integrate._ivp.ivp.OdeResult,
            K: float,
            proportions: bool,
            derivative: bool,
            logder: bool) -> None:

        if self.equation == 'one':
            Y = arr.y[3]
        elif self.equation == 'two':
            Y = arr.y[2]
        else:
            Y = arr.y[-1]
        
        deriv = np.diff(Y)
        x = arr.y[0]

        maximum = np.max(deriv)
        max_ = np.argmax(deriv)
        
        if not derivative:
            if proportions:
                x_ = x[:-1]/K
                thr = x[max_]/K
                banner = 'Population/K'
            else:
                x_ = x[:-1]
                thr = x[max_]
                banner = 'Population'

            plt.plot(x_, deriv)
            plt.xlabel(banner)
            plt.axvline(thr, ls='--', c='g')
            plt.axhline(maximum, ls='--', c='r')
            plt.ylabel('Mortality function')
            plt.title(r'$\mu(X,t)$')
            plt.grid(True)
            plt.show()

        else:
            der = np.diff(deriv)
            der = savgol_filter(der, 550, 5)
            max_der = np.max(der)
            max_mom = np.argmax(der)

            if proportions:
                x_ = x[:-2]/K
                thr = x[max_]/K
                thr_ = x[max_mom]/K
                banner = 'Population/K'
            else:
                x_ = x[:-2]
                thr = x[max_]
                thr_ = x[max_mom]
                banner = 'Population'

            _, ax = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)

            ax[0].plot(x_, deriv[:-1])

            if not logder:
                ax[1].plot(x_, der)
            else:
                ax[1].semilogy(x_, der)

            ax[0].set_title('Mortality function', fontsize = 18)
            ax[1].set_title('Mortality function derivative', fontsize = 18)

            ax[0].set_xlabel(banner, fontsize = 16)
            ax[1].set_xlabel(banner, fontsize = 16)

            ax[0].set_ylabel(r'$\mu(X,t)$', fontsize = 20)
            ax[1].set_ylabel(r'$\frac{\partial \mu(X,t)}{ \partial X}$', fontsize = 24)

            ax[0].axvline(thr, color='r', ls='--')
            ax[1].axvline(thr_, color='r', ls='--')
            ax[1].axhline(max_der, ls='-.', color='g')
            ax[0].axhline(maximum, ls='-.', color='g')
            ax[0].grid(True)
            ax[1].grid(True)

            print('-'*50)
            print('-'*50)
            print('Maximum value of mortality function derivative:', np.round(max_der))
            print('-'*50)
            print('Population value for mortality function derivative maximum (proportional to K):', np.round(thr_, 2))

        if not derivative:
            print('-'*50)

        print('-'*50)
        print('Maximum value of mortality function:', np.round(maximum))
        print('-'*50)
        print('Population value for mortality function maximum (proportional to K):', np.round(x[max_]/K, 2))
    
    def _plot_stable_points(self, root: int, proportions: bool, plot_thr: bool) -> None:
        if self.include:
            raise ValueError('There is no analytical solution for this equation -> there are no stable points')
        t = self.t
        res = np.zeros(len(t))
        b, K, _, r, _, a, _, _, _, _ = tuple(self.conf_.values()) if not self.custom else tuple(self.custom_conf.values())
        thr = self.threshold if not self.custom else self.custom_thr
        for i in range(len(t)):
            if root == 1:
                res[i] = 0.5*(K*a**2*b*t[i]**2 + 2.0*r)/(a**2*b*t[i]**2) - 1.4142135623731\
                         * np.sqrt(0.125*K**2*a**4*b**2*t[i]**4 + K*a**3*b*t[i]**2 - 0.5*K*a**2*b*r*t[i]**2 + 0.5*r**2) \
                         / (a**2*b*t[i]**2)
            elif root == 2:
                res[i] = 0.5*(K*a**2*b*t[i]**2 + 2.0*r)/(a**2*b*t[i]**2) + 1.4142135623731\
                         * np.sqrt(0.125*K**2*a**4*b**2*t[i]**4 + K*a**3*b*t[i]**2 - 0.5*K*a**2*b*r*t[i]**2 + 0.5*r**2) \
                         / (a**2*b*t[i]**2)
        
        if proportions:
            res = res/K
            res_ = self.calculate_population().y[0]/K
            label = 'Population/K'
        else:
            thr = thr*K
            label = 'Population'
            res_ = self.calculate_population().y[0]
        plt.plot(t*self.coeff, res, label=f'Stable points root {root}')
        plt.plot(t*self.coeff, res_, label='Numerical solution of DE', ls='--')
        plt.xlabel('Years')
        plt.ylabel(label)
        if plot_thr:
            plt.axhline(thr, ls='--', c='r')
        plt.legend()
        plt.grid(True)
        plt.show()

    def variator(
            self,
            fraction: float = 5,
            sampling_freq: int = 4,
            x_bound: float = 300,
            only_z: bool = False,
            only_r: bool = False,
            only_sigma: bool = False,
            only_alpha: bool = False,
            only_d: bool = False,
            only_g: bool = False,
            z_min: float = 0.,
            z_max: float = 1.0,
            d_min: float = 0.1,
            d_max: float = 1.0,
            legend: bool = True,
            proportions: bool = True,
            minimum: float = None,
            maximum: float = None) -> None:
        """
            Perturb the parameters of a system to see the results.

            ######
            Args: fraction for interval as {parameter/fraction; parameter*fraction},
                  sampling_freq - amount of equidistant points to separate the interval,
                  x_bound = cut the plot on this value of time, only_* - variate only * parameter,
                  {z_min; z_max} and {d_min;d_max} - bounds for proportion of alive mutants and their death rate,
                  legend - show legend on plot or not, proportions - whether on not to plot population as a fraction of K.

            ######
            Output: plots and lifespans
        """

        if not self.custom:
            init = self.init
            thr = self.threshold
            K = self.conf_['K']
            M = self.conf_['M']
            config_ = self.conf_
        else:
            init = self.custom_init
            thr = self.custom_thr
            config_ = self.custom_conf
            K = config_['K']
            M = config_['M']
        
        if only_r:
            self._vary_parameter('r', fraction, sampling_freq, init,
                                 thr, x_bound, legend, config_, K, proportions, minimum, maximum)
        elif only_alpha:
            self._vary_parameter('alpha', fraction, sampling_freq, init,
                                 thr, x_bound, legend, config_, K, proportions, minimum, maximum)
        elif only_sigma:
            self._vary_parameter('sigma', fraction, sampling_freq, init,
                                 thr, x_bound, legend, config_, K, proportions, minimum, maximum)
        elif only_z:
            self._vary_z_parameter(z_min, z_max, sampling_freq, init,
                                   thr, x_bound, legend, config_, only_d, K, proportions)
        elif only_d:
            self._vary_z_parameter(z_min, z_max, sampling_freq, init,
                                   thr, x_bound, legend, config_, only_d, K, proportions)
        elif only_g:
            self._vary_g_parameter( fraction, sampling_freq, init, thr, x_bound,
                                    legend, config_, K, M, proportions, minimum, maximum)
        else:
            self._variator(fraction, sampling_freq, config_, init, x_bound, legend,
                           thr, z_min, z_max, d_min, d_max, K, proportions)

    def _variator(
            self, 
            fraction: float,
            sampling_freq: int,
            config: dict,
            init: dict,
            x_bound: float,
            legend: bool,
            thr: float,
            z_min: float,
            z_max: float,
            d_min: float,
            d_max: float,
            K: float,
            proportion: bool) -> None:
        
        if (self.equation == 'one') & (not self.include):
            param_ind = ['sigma', 'alpha', 'r']
            sols = [[], [], []]
            print('No alive mutants and Model 1 is used. Varying only sigma, alpha and r.')
        else:
            param_ind = ['sigma', 'alpha', 'beta', 'eps'] if self.equation == 'two' else ['sigma', 'alpha', 'z', 'theta']
            sols = [[], [], [], []]

        mantissa = {'sigma': 4, 'alpha': 13, 'beta': 13, 'eps': 3, 'theta': 2, 'z': 2, 'r': 4}
        names = {
            'sigma': 'sigma', 'alpha': 'alpha', 'beta': 'beta', 'eps': 'epsilon',
            'theta': 'alive mutants death rate', 'z': 'proportion of alive mutants', 'r': 'somatic cells recovery rate'
        }
        ranges = []

        for i in param_ind:
            if i == 'theta':
                ranges.append(np.linspace(d_min, d_max, sampling_freq))
            elif i == 'z':
                ranges.append(np.linspace(z_min, z_max, sampling_freq))
            else:
                ranges.append(np.linspace(config[i]/fraction, config[i]*fraction, sampling_freq))

        for n in range(len(sols)):
            for param_value in ranges[n]:
                conf_ = config.copy()
                conf_[param_ind[n]] = param_value

                sol = solve_ivp(
                    self.model,
                    t_span=(self.start, self.end),
                    y0=list(init.values()),
                    t_eval=self.t,
                    method=self.method,
                    args=tuple(conf_.values()),
                    dense_output=False, 
                    atol=1e-6, 
                    rtol=1e-6)
                
                sols[n].append(sol)

        if (self.equation == 'one') & (not self.include):
            _, axs = plt.subplots(1, 3, figsize=(25, 8))
        else:
            _, axs = plt.subplots(2, 2, figsize=(20, 14))
        axs = axs.flatten()

        for i in range(len(axs)):
            for j in range(len(sols[i])):

                axs[i].plot(
                    self.t*self.coeff, sols[i][j].y[0] if not proportion else sols[i][j].y[0]/K, 
                    label=f'Somatic population for {names[param_ind[i]]} = {np.round(ranges[i][j], mantissa[param_ind[i]])}')
                
            axs[i].grid(True)
            axs[i].set_xlabel('Years')
            axs[i].set_ylabel('Population')
            axs[i].set_xlim(0, x_bound)
            if proportion: 
                axs[i].axhline(thr, ls='--', color='r')
            else:
                axs[i].axhline(thr*K, ls='--', color='r')
            axs[i].set_title(f'Population number for different {names[param_ind[i]]}')
            if legend:
                axs[i].legend(loc='upper right')

    def _vary_parameter(
            self, 
            param_name: str,
            fraction: float,
            sampling_freq: int,
            initial_conditions: dict,
            threshold: float,
            x_bound: float,
            legend: bool,
            config: dict,
            K: float,
            proportion: bool,
            minimum: float,
            maximum: float) -> None:
        
        if (minimum is not None) & (maximum is not None):
            param_range = np.linspace(minimum, maximum, sampling_freq)
        else:
            param_range = np.linspace(config[param_name] / fraction, fraction * config[param_name], sampling_freq)
        solutions = []

        for param_value in param_range:
            conf = config.copy()
            conf[param_name] = param_value

            solution = solve_ivp(
                self.model,
                t_span=(self.start, self.end), 
                y0=list(initial_conditions.values()), 
                t_eval=self.t, 
                method=self.method, 
                args=tuple(conf.values()), 
                dense_output=False, 
                atol=1e-6, 
                rtol=1e-6)
            
            solutions.append(solution)

        print('Lower bound result:')
        _ = self.lifespan(custom_solution=solutions[0])
        print('-'*40)
        print('Upper bound result:')
        _ = self.lifespan(custom_solution=solutions[-1])

        self._plot_variation(solutions, param_range, param_name, threshold, x_bound, legend, K, proportion)
    
    def _vary_z_parameter(
            self, 
            z_min: float,
            z_max: float,
            steps: int,
            initial_conditions: dict,
            threshold: float,
            x_bound: float,
            legend: bool,
            config: dict,
            only_d: bool,
            K: float,
            proportion: bool) -> None:
        
        if self.equation == 'two':
            raise ValueError('No alive mutants in two equation model. Consider using Model 1 equation.')
        
        param_name = 'alive mutants proportion' if not only_d else 'alive mutants death rate'
    
        z_range = np.linspace(z_min, z_max, steps)
        solutions = []
    
        for z_value in z_range:
            conf = config.copy()
            if not only_d:
                conf['z'] = z_value
            else:
                conf['theta'] = z_value

            solution = solve_ivp(
                self.model,
                t_span=(self.start, self.end), 
                y0=list(initial_conditions.values()), 
                t_eval=self.t, 
                method=self.method, 
                args=tuple(conf.values()), 
                dense_output=False, 
                atol=1e-6, 
                rtol=1e-6)
            
            solutions.append(solution)

        print('Lower bound result:')
        _ = self.lifespan(custom_solution=solutions[0])
        print('-'*40)
        print('Upper bound result:')
        _ = self.lifespan(custom_solution=solutions[-1])

        self._plot_variation(solutions, z_range, param_name, threshold, x_bound, legend, K, proportion)

    def _vary_g_parameter(self,
             fraction: float,
             sampling_freq: int,
             initial_conditions: dict,
             threshold: float,
             x_bound: float,
             legend: bool,
             config: dict,
             K: float,
             M: float,
             proportion: bool,
             minimum: float,
             maximum: float) -> None:

        if self.equation == 'one':
            raise ValueError('No \'g\' parameter in Model 1')
        else:
            if (minimum is None) or (maximum is None):
                space = np.linspace(config['g']/fraction, config['g']*fraction, sampling_freq)
            else:
                space = np.linspace(minimum, maximum, sampling_freq)

            solutions = []

            for g in space:
                conf = config.copy()
                conf['g'] = g

                solution = solve_ivp(
                    self.model,
                    t_span=(self.start, self.end),
                    y0=list(initial_conditions.values()),
                    t_eval=self.t,
                    method=self.method,
                    args=tuple(conf.values()),
                    dense_output=False,
                    atol=1e-6,
                    rtol=1e-6)

                solutions.append(solution)

            print('Lower bound result:')
            _ = self.lifespan(custom_solution=solutions[0])
            print('-' * 40)
            print('Upper bound result:')
            _ = self.lifespan(custom_solution=solutions[-1])

            _, ax = plt.subplots(1, 2, figsize=(12, 8))

            if proportion:
                divisor1, divisor2 = K, M
                thr1, thr2 = threshold, threshold
            else:
                divisor1, divisor2 = 1, 1
                thr1, thr2 = threshold*config['K'], threshold*config['M']

            for solution, value in zip(solutions, space):
                ax[0].plot(self.t * self.coeff, solution.y[0] / divisor1,
                           label=f'Somatic population for g={np.round(value, 6)}')
                ax[1].plot(self.t * self.coeff, solution.y[1] / divisor2,
                           label=f'Stem cells population for g={np.round(value, 6)}')

            ax[0].set(xlim=(0, x_bound), xlabel='Years', ylabel='Population')
            ax[1].set(xlim=(0, x_bound), xlabel='Years', ylabel='Population')

            ax[0].axhline(thr1, c='r', ls='--')
            ax[1].axhline(thr2, c='r', ls='--')

            if legend:
                ax[0].legend(loc='upper right')
                ax[1].legend(loc='upper right')

    def _plot_variation(
            self,
            solutions: list,
            param_range: np.ndarray,
            param_name: str,
            threshold: float,
            x_bound: float,
            legend: bool,
            K: float,
            proportion: bool) -> None:

        fig, ax = plt.subplots(figsize=(12, 8))

        for solution, param_value in zip(solutions, param_range):
            ax.plot(self.t * self.coeff, solution.y[0] if not proportion else solution.y[0]/K,
                    label=f'{param_name} = {np.round(param_value, 13)}')

        self._finalize_plot(ax, param_name, threshold, x_bound, legend, K, proportion)

    @staticmethod
    def _finalize_plot(
            ax,
            param_name: str,
            threshold: float,
            x_bound: float,
            legend: bool,
            K: float,
            proportion: bool) -> None:

        ax.grid(True)
        ax.set_xlabel('Years')
        ax.set_ylabel('Population')
        ax.set_title(f'Population for different {param_name}')
        ax.set_xlim(0, x_bound)
        if proportion: 
            ax.axhline(threshold, ls='--', color='r')
        else: 
            ax.axhline(threshold*K, ls='--', color='r')
        if legend:
            ax.legend(loc='upper right')
        plt.show()
