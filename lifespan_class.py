import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style
from scipy.integrate import solve_ivp
import numpy as np
import warnings
from typing import List, Dict

class Somatic_LS(object):
    def __init__(
        self, 
        organ:str='liver', 
        method:str='RK45', 
        start_time:float=0, 
        end_time:float=300, 
        include_mutants:bool=False, 
        equation:str='single',
        print_config:bool=True,
        custom_conf:List = None,
        custom_thr:float = None,
        custom_init:List = None,
        style:str = 'bmh',
        print_methods:bool = False,
        print_styles:bool = False):

        '''
            ####Class for lifespan simulation based on the only hallmark of aging - somatic mutations.

            ######
            Args: organ for simulation, method for solve_ivp, 
            start and end times in years, include alive mutants in single-equation system or not, 
            type of system, whether or not to print config, custom_* - specify your custom parameters fot a system.

            ######
            Output: populations, plots and variations.
        '''
        if print_styles:
            print(matplotlib.style.available)
        plt.style.use(style)
        if print_methods:
            print('RK45, RK23, Radau, DOP853, LSODA, BDF')
        organs = ['liver', 'mouse liver', 'lungs', 'spinal cord']
        self.method = method
        self.organ = organ
        self.custom = False
        if self.organ not in organs:
            print('This organ is not specified. Use your custom config, threshold and initial conditions to solve the system')
            if (custom_conf is None) or (custom_thr is None) or (custom_init is None):
                raise ValueError('Please specify your parameters, initial conditions and threshold value when using not built-in organs.')
            else:
                self.custom_conf = custom_conf
                self.custom_thr = custom_thr
                self.custom_init = custom_init
                self.custom = True
        self.conf = self._set_config()
        if not include_mutants:
            self.conf[-1] = 0
            self.conf[-2] = 0

        coeff = 21.5 / (365.25 * 24)
        self.start = int(start_time / coeff)
        self.end = int(np.round(end_time / coeff))
        self.span = -self.start + self.end
        self.t = np.linspace(self.start, self.end, self.end - self.start)
        self.include = include_mutants
        self.equation = equation

        if method != 'RK45':
            warnings.warn(f'Using {method} as a solver for solve_ivp. Consider using RK45 as the most stable solver')
        
        if (self.organ == 'spinal cord')&(self.equation == 'two'):
            print('======= No two-equation system for spinal cord. Changing to single-equation. =======')
            self.equation = 'single'

        if print_config: self._print_config()

    def _print_config(self) -> None:
        print(f'''
              CONFIG FOR SIMULATION HAS BEEN CREATED
              {'-'*40}
              Final parameters set:
              --organ: {self.organ},
              --start: {np.round(self.start*(21.5 / (365.25 * 24)))} years,
              --end: {np.round(self.end*(21.5 / (365.25 * 24)))} years,
              --type of system: {self.equation} equation system,
              --solver method: {self.method},
              --include mutants: {self.include}
              {'-'*40}
              ''')
    
    def _set_config(self) -> List:
        configs = {
            'liver': [0.087, 2e11, 2e11/94000, 4/407, 0.064, (3.5e-9)*9.6e-3, (1.83e-9)*9.6e-3, 4/407, 0.9, 0.239],
            'mouse liver': [0.087, 3.37e8, 3.37e8/94000, 63/407, 0.064, 35*(3.5e-9), 35*(1.83e-9), 63/407, 0.9, 0.239],
            'lungs': [0.073, 10.5e9, 0.07*10.5e9, 0.001/407, 0.007, 6.392476819356688e-12, 6.392476819356688e-12 / 1.9126, 0.001/407, 0.9, 0.239],
            'spinal cord': [0.085, 222e6, 0, 0, 0, 0.9047619*(3.5e-9)*0.0013563, 0, 0, 0.9, 0.239]
        }
        return configs.get(self.organ, configs['liver'])
    
    def _model_one(self, t, y, s, K, M, r, e, a, b, g, z, d) -> List:
        X, C, F, m = y
        m1 = 0.5 * s * (1 - X / K)
        dXdt = r * X * (1 - X / K) - a * X - m1 * ((C + (1 - z) * a * X * t + d * C * t) ** 2)
        dCdt = r * C * (1 - C / K) + z * a * X - d * C
        dFdt = (1 - z) * a * X + d * C
        dmdt = m1 * ((C + (1 - z) * a * X * t + d * C * t) ** 2)
        return [dXdt, dCdt, dFdt, dmdt]

    def _model_two(self, t, y, s, K, M, r, e, a, b, g, z, d) -> List:
        X, Y, m = y
        m1 = 0.5 * s * (1 - (X + Y) / (K + M))
        dXdt = r * X * (1 - X / K) + 2 * e * Y - a * X - m1 * ((a * X + b * Y) ** 2 * t ** 2)
        dYdt = g * Y * (1 - Y / M) - e * Y - b * Y
        dmdt = 0.5 * s * (1 - (X + Y) / (K + M)) * ((a * X + b * Y) ** 2 * t ** 2)
        return [dXdt, dYdt, dmdt]
    
    def get_initial_conditions(self) -> List:
        K = self.conf[1]
        M = self.conf[2]
        if self.equation == 'single':
            initial = {
                'liver': [0.31 * K, 0, 0, 0],
                'lungs': [0.78 * K, 0, 0, 0],
                'spinal cord': [K, 0, 0, 0],
                'mouse liver': [0.31 * K, 0, 0, 0]
            }
        elif self.equation == 'two':
            initial = {
                'liver': [0.31 * K, 0.31 * M, 0],
                'lungs': [0.78 * K, 0.78 * M, 0],
                'mouse liver': [0.31 * K, 0.31 * M, 0]
            }

        return initial.get(self.organ, initial['liver'])
    
    def get_model(self):
        if self.equation == 'single':
            return self._model_one
        elif self.equation == 'two':
            return self._model_two

    def calculate_population(self):
        '''
            ####Calculate populations by solving differential equation system numerically.

            ######
            Args: self

            ######
            Output: solve_ivp object with calculated cell populations
        '''
        if not self.custom:
            conf = tuple(self.conf)
            initial = self.get_initial_conditions()
        else:
            conf = tuple(self.custom_conf)
            initial = self.custom_init
        model = self.get_model()
        solution = solve_ivp(model,t_span=(self.start, self.end), y0=initial, t_eval=self.t, method = self.method,args=conf, dense_output=True)
        self.sol = solution
        return solution  
    
    def get_threshold(self) -> Dict:
        thr = {
            'liver': 0.3,
            'mouse liver': 0.3,
            'lungs': 0.4,
            'spinal cord': 0.9
        }
        self.thr = thr.get(self.organ, thr['liver'])
        return thr.get(self.organ, thr['liver'])

    def lifespan(
        self, 
        custom_solution = None) -> int:

        '''
            ####Calculate a lifespan of somatic population given a cutoff value.

            ######
            Args: custom_solution - your own solve_ivp object

            ######
            Output: a moment of time when a population reaches a cutoff value.
        '''
        if self.custom:
            thr = self.custom_thr
            K = self.custom_conf[1]
        else:
            thr = self.get_threshold()
            K = self.conf[1]

        if custom_solution is None:
            arr = self.calculate_population()
        else:
            arr = custom_solution

        l = []
        k = []
        for i in range(self.span):
            if arr.y[0][i]/K <= thr:
                l.append(i)
                break
        for i in range(self.span):
            if arr.y[0][i]/K > 0.99:
                k.append(i)
                break
        if len(l) != 0:
            if self.organ == 'mouse liver':
                print('Life expentancy (years):', np.round((l[0] * 21.5)/(365.25 * 24), 2))
            else:
                print('Life expentancy (years):', np.round((l[0] * 21.5)/(365.25 * 24)))
            print('-'*50)
            ls = l[0]
        else:
            print('Havent died')
            print('-'*50)
            ls = []

        if len(k) != 0:
            if arr.y[0][0] == K:
                print('No resection')
            else:
                print('Time of regeneration of somatic cells (years):', np.round((k[0] * 21.5)/(365.25 * 24), 2))
        else:
            if arr.y[0][0] != K:
                print('Havent regenerated')
        
        return ls
    
    def plot_curves(
        self, 
        population:str = 'somatic', 
        view_all:bool = False, 
        proportions:bool = True, 
        plot_thr:bool = True) -> None:

        '''
            ####Plot results of simulation.

            ######
            Args: population - type of population, view_all - show only till the moment of death or not, 
            proportions - plot as population/(population limit), plot_thr - plot cutoff value

            ######
            Output: matplotlib.pyplot plots, lifespan
        '''

        if not self.custom:
            thr = self.get_threshold()
            K = self.conf[1]
        else:
            thr = self.custom_thr
            K = self.custom_conf[1]
        life = self.lifespan()
        arr = self.calculate_population()

        if self.equation == 'single':
            if population == 'somatic':
                Y = arr.y[0]
            elif population == 'alive mutants':
                Y = arr.y[1]
            elif population == 'dead mutants':
                Y = arr.y[2]
            elif (population == 'mortality function'):
                self._plot_mortality(arr,life)
                return None
            
        elif self.equation == 'two':
            if population == 'somatic':
                Y = arr.y[0]
            elif population == 'stem':
                Y = arr.y[1]
            elif population == 'mortality function':
                self._plot_mortality(arr,life)
                return None
        
        if proportions:
            plt.plot(self.t*21.5/(365.25*24), Y/K, label = population)
            if plot_thr:
                plt.axhline(thr, ls = '--', color = 'r')
        else:
            plt.plot(self.t*21.5/(365.25*24), Y, label = population)
            if plot_thr:
                plt.axhline(thr*K, ls = '--', color = 'r')
        plt.xlabel('Years')
        plt.ylabel('Population')
        if not view_all:
            plt.xlim(self.start, life*21.5/(365.25*24))
        plt.legend()
        plt.grid('True')
        plt.show()
    
    def _plot_mortality(self, arr, life:int) -> None:
        if self.equation == 'single':
            Y = arr.y[3]
        else:
            Y = arr.y[2]
        deriv = np.diff(Y)

        maximum = np.argmax(deriv)
        max_der = np.max(deriv)

        _, axes = plt.subplots(1, 2, figsize = (12,8))

        axes[0].plot(self.t * 21.5/(365.25 * 24), Y)
        axes[1].plot(self.t[:-1] * 21.5/(365.25 * 24), deriv)

        axes[0].grid('True')
        axes[1].grid('True')

        axes[1].set_xlabel('Years')
        axes[1].set_ylabel('dM/dt')

        axes[0].set_xlabel('Years')
        axes[0].set_ylabel('M(t)')

        axes[0].set_title('Mortality function')
        axes[1].set_title('Mortality function derivative')

        axes[0].axvline(maximum * 21.5/(365.25 * 24), color = 'r', ls = '--')
        axes[1].axvline(maximum * 21.5/(365.25 * 24), color = 'r', ls = '--')
        axes[1].axhline(max_der, ls = '-.', color = 'g')

        print('Max:', np.round(maximum * 21.5/(365.25 * 24), 0))
        print('Max deriv:', np.round(max_der, 0))
        print('Ratio of max deriv to total lifespan in %:', np.round((maximum * 21.5/(365.25 * 24))/((life * 21.5)/(365.25 * 24)) * 100, 1))
    
    def variator(
        self, 
        fraction:float = 5, 
        sampling_freq:int = 4, 
        x_bound:float = 300, 
        only_z:bool=False, 
        only_r:bool=False, 
        only_sigma:bool=False, 
        only_alpha:bool=False, 
        only_d:bool=False,
        z_min:float=0.1, 
        z_max:float=0.9,
        d_min:float=0.1,
        d_max:float=0.9, 
        legend:bool=True,
        proportions:bool=True) -> None:

        '''
            ####Perturb the parameters of a system to see the results.

            ######
            Args: fraction for interval as {parameter/fraction; parameter*fraction}, sampling_freq - amount of equidistant points to separate the interval, 
            x_bound = cut the plot on this value of time, only_* - variate only * parameter, {z_min; z_max} and {d_min;d_max} - bounds for proportion of alive mutants and their death rate,
            legend - show legend on plot or not, proportions - whether on not to plot population as a fraction of K.

            ######
            Output: plots and lifespans
        '''
        
        if not self.custom:
            init = self.get_initial_conditions()
            thr = self.get_threshold()
            K = self.conf[1]
            config_ = self.conf
        else:
            init = self.custom_init
            thr = self.custom_thr
            config_ = self.custom_conf
            K = config_[1]
        model = self.get_model()
        
        if only_r:
            self._vary_parameter('r', 3, fraction, sampling_freq, model, init, thr, x_bound, legend,config_,K, proportions)
        elif only_alpha:
            self._vary_parameter('alpha', 5, fraction, sampling_freq, model, init, thr, x_bound, legend, config_, K, proportions)
        elif only_sigma:
            self._vary_parameter('sigma', 0, fraction, sampling_freq, model, init, thr, x_bound, legend, config_, K, proportions)
        elif only_z:
            self._vary_z_parameter(z_min, z_max, sampling_freq, model, init, thr, x_bound, legend, config_, only_d, K, proportions)
        elif only_d:
            self._vary_z_parameter(z_min, z_max, sampling_freq, model, init, thr, x_bound, legend, config_, only_d, K, proportions)
        else:
            self._variator(fraction, sampling_freq, model, config_, init, x_bound, legend, thr, z_min, z_max, d_min, d_max, K, proportions)   

    def _variator(self, fraction, sampling_freq, model, config, init, x_bound, legend, thr, z_min, z_max, d_min, d_max, K, proportion):
        if (self.equation == 'single')&(self.include == False):
            param_ind = [0,5,3]
            sols = [[], [], []]
            print('No alive mutants and Model 1 is used. Varying only sigma, alpha and r.')
        else:
            param_ind = [0, 5, 6, 4] if self.equation=='two' else [0,5,-1,-2]
            sols = [[], [], [], []]
        mantissa = {0:4, 5:13, 6:13, 4:3, -1:2, -2:2, 3: 4}
        names = {0: 'sigma', 5: 'alpha', 6: 'beta', 4: 'epsilon', -1:'alive mutants death rate', -2:'proportion of alive mutants', 3: 'somatic cells recovery rate'}
        ranges = []
        for i in param_ind:
            if i == -1:
                ranges.append(np.linspace(d_min, d_max, sampling_freq))
            elif i == -2:
                ranges.append(np.linspace(z_min, z_max, sampling_freq))
            else:
                ranges.append(np.linspace(config[i]/fraction, config[i]*fraction, sampling_freq))
        for n in range(len(sols)):
            for param_value in ranges[n]:
                conf_ = config.copy()
                conf_[param_ind[n]] = param_value
                sol = solve_ivp(model, t_span = (self.start, self.end), y0 = init, t_eval = self.t, method = self.method, args = tuple(conf_), dense_output=True)
                sols[n].append(sol)
        if (self.equation == 'single')&(self.include == False):
            fig, axs = plt.subplots(1, 3, figsize = (25, 8))
        else:
            fig, axs = plt.subplots(2,2, figsize = (20,14))
        axs_ = axs.flatten()
        for i in range(len(axs_)):
            for j in range(len(sols[i])):
                axs_[i].plot(self.t*21.5/(365.25*24), sols[i][j].y[0] if not proportion else sols[i][j].y[0]/K, label = f'Somatic population for {names[param_ind[i]]} = {np.round(ranges[i][j], mantissa[param_ind[i]])}')
            axs_[i].grid('True')
            axs_[i].set_xlabel('Years')
            axs_[i].set_ylabel('Population')
            axs_[i].set_xlim(0,x_bound)
            if proportion: 
                axs_[i].axhline(thr, ls = '--', color = 'r')
            else:
                axs_[i].axhline(thr*K, ls = '--', color = 'r')
            axs_[i].set_title(f'Population number for different {names[param_ind[i]]}')
            if legend:
                axs_[i].legend(loc = 'upper right')

    def _vary_parameter(self, param_name, param_index, fraction, sampling_freq, model, initial_conditions, threshold, x_bound, legend, config, K, proportion):
        param_range = np.linspace(config[param_index] / fraction, fraction * config[param_index], sampling_freq)
        solutions = []

        for param_value in param_range:
            conf = config.copy()
            conf[param_index] = param_value
            solution = solve_ivp(model, t_span=(self.start, self.end), y0=initial_conditions, t_eval=self.t, method=self.method, args=tuple(conf), dense_output=True)
            solutions.append(solution)

        print('Lower bound result:')
        ls_ = self.lifespan(custom_solution=solutions[0])
        print('-'*40)
        print('Upper bound result:')
        ls = self.lifespan(custom_solution=solutions[-1])

        self._plot_variation(solutions, param_range, param_name, threshold, x_bound, legend, K, proportion)
    
    def _vary_z_parameter(self, z_min, z_max, steps, model, initial_conditions, threshold, x_bound, legend, config, only_d, K, proportion):
        if self.equation == 'two':
            raise ValueError('No alive mutants in two equation model. Consider using single equation.')
        
        param_name = 'alive mutants proportion' if not only_d else 'alive mutants death rate'
    
        z_range = np.linspace(z_min, z_max, steps)
        solutions = []
    
        for z_value in z_range:
            conf = config.copy()
            if not only_d:
                conf[-2] = z_value
            else:
                conf[-1] = z_value
            solution = solve_ivp(model, t_span=(self.start, self.end), y0=initial_conditions, t_eval=self.t, method=self.method, args=tuple(conf), dense_output=True)
            solutions.append(solution)

        print('Lower bound result:')
        ls_ = self.lifespan(custom_solution=solutions[0])
        print('-'*40)
        print('Upper bound result:')
        ls = self.lifespan(custom_solution=solutions[-1])
        self._plot_variation(solutions, z_range, param_name, threshold, x_bound, legend, K, proportion)
    
    def _plot_variation(self, solutions, param_range, param_name, threshold, x_bound, legend, K, proportion):
        fig, ax = plt.subplots(figsize=(12, 8))

        for solution, param_value in zip(solutions, param_range):
            ax.plot(self.t * 21.5 / (365.25 * 24), solution.y[0] if not proportion else solution.y[0]/K, label=f'{param_name} = {np.round(param_value, 13)}')

        self._finalize_plot(ax, param_name, threshold, x_bound, legend, K, proportion)
        
    def _finalize_plot(self, ax, param_name, threshold, x_bound, legend, K, proportion):
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