import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import warnings
from typing import List, Tuple, Dict

plt.style.use("bmh")

class Somatic_LS(object):
    def __init__(
        self, 
        organ:str='liver', 
        method:str='RK45', 
        start_time:float=0, 
        end_time:float=300, 
        include_mutants:bool=False, 
        equation:str='single',
        print_config:bool=True):

        '''
            Class for lifespan simulation based on the only hallmark of aging - somatic mutations.

            ######
            Args: organ for simulation, method for solve_ivp, start and end times in years, include alive mutants in single-equation system or not, type of system, whether or not to print config.

            ######
            Output: populations, plots and variations.
        '''

        organs = ['liver', 'mouse liver', 'lungs', 'spinal cord']
        self.method = method
        self.organ = organ
        if self.organ not in organs:
            print('This organ is not specified. Use your custom config, threshold and initial conditions to solve the system')
            print('Fist you need calculate_population, then lifespan')
        self.conf = self.set_config()
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
            print('=======No two-equation system for spinal cord. Changing to single-equation.=======')
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
    
    def set_config(self) -> List:
        configs = {
            'liver': [0.087, 2e11, 2e11/94000, 4/407, 0.064, (3.5e-9)*9.6e-3, (1.83e-9)*9.6e-3, 4/407, 0.9, 0.239],
            'mouse liver': [0.087, 3.37e8, 3.37e8/94000, 63/407, 0.064, 35*(3.5e-9), 35*(1.83e-9), 63/407, 0.9, 0.239],
            'lungs': [0.073, 10.5e9, 0.07*10.5e9, 0.001/407, 0.007, 6.392476819356688e-12, 6.392476819356688e-12 / 1.9126, 0.001/407, 0.9, 0.239],
            'spinal cord': [0.085, 222e6, 0, 0, 0, 0.9047619*(3.5e-9)*0.0013563, 0, 0, 0.9, 0.239]
        }
        return configs.get(self.organ, configs['liver'])
    
    def model_one(self, t, y, s, K, M, r, e, a, b, g, z, d) -> List:
        X, C, F, m = y
        m1 = 0.5 * s * (1 - X / K)
        dXdt = r * X * (1 - X / K) - a * X - m1 * ((C + (1 - z) * a * X * t + d * C * t) ** 2)
        dCdt = r * C * (1 - C / K) + z * a * X - d * C
        dFdt = (1 - z) * a * X + d * C
        dmdt = m1 * ((C + (1 - z) * a * X * t + d * C * t) ** 2)
        return [dXdt, dCdt, dFdt, dmdt]

    def model_two(self, t, y, s, K, M, r, e, a, b, g, z, d) -> List:
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
                'liver': [0.31*K, 0, 0, 0],
                'lungs': [0.78*K, 0, 0, 0],
                'spinal cord': [K, 0, 0, 0],
                'mouse liver': [0.31*K, 0, 0, 0]
            }
        elif self.equation == 'two':
            if self.organ == 'spinal cord':
                raise ValueError('No two-equation system for spinal cord. Consider using single equation')
            initial = {
                'liver': [0.31*K, 0.31*M, 0],
                'lungs': [0.78*K, 0.78*M, 0],
                'mouse liver': [0.31*K, 0.31*M, 0]
            }
        self.init = initial.get(self.organ, initial['liver'])
        return initial.get(self.organ, initial['liver'])
    
    def get_model(self):
        if self.equation == 'single':
            return self.model_one
        elif self.equation == 'two':
            if self.organ == 'spinal cord':
                raise ValueError('No two-equation system for spinal cord. Consider using single equation')
            return self.model_two

    def calculate_population(self, custom_conf:List=None, custom_init:List = None):
        '''
            Calculate populations by solving differential equation system numerically.

            ######
            Args: custom_conf - whether or not to use custom set of parameters, custom_init - custom initial conditions

            ######
            Output: solve_ivp object with calculated cell populations
        '''
        if (custom_conf == None)&(custom_init == None):
            conf = tuple(self.conf)
            initial = self.get_initial_conditions()
        else:
            conf = tuple(custom_conf)
            initial = custom_init
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
        custom_solution = None,
        custom_thr:float = None,
        custom_conf:List = None) -> int:

        '''
            Calculate a lifespan of somatic population given a cutoff value.

            ######
            Args: custom_solution - your own solve_ivp object, custom_thr - your own cutoff value, custom_conf - custom set of parameters.

            ######
            Output: a moment of time when a population reaches a cutoff value.
        '''

        if custom_conf is None:
            K = self.conf[1]
        else:
            K = custom_conf[1]

        if custom_solution is not None:
            arr = custom_solution
        else:
            arr = self.calculate_population()
        
        if custom_thr is None:
            thr = self.get_threshold()
        else:
            thr = custom_thr

        #l = [i for i in range(self.span) if arr.y[0][i]/K <= thr]
        #k = [i for i in range(self.span) if arr.y[0][i]/K > 0.99]
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
                print('Life expentancy (years):', np.round((l[0]*21.5)/(365.25*24), 2))
            else:
                print('Life expentancy (years):', np.round((l[0]*21.5)/(365.25*24)))
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
                print('Time of regeneration of somatic cells (years):', np.round((k[0]*21.5)/(365.25*24), 2))
        else:
            if arr.y[0][0] != K:
                print('Havent regenerated')
        
        self.ls = ls
        return ls
    
    def plot_curves(
        self, 
        population:str = 'somatic', 
        view_all:bool = False, 
        proportions:bool = True, 
        plot_thr:bool = True,
        custom_conf:List = None,
        custom_thr:float = None,
        custom_sol = None,
        custom_ls:int = None) -> None:

        '''
            Plot results of simulation.

            ######
            Args: population - type of population, view_all - show only till the moment of death or not, proportions - plot as population/(population limit), plot_thr - plot cutoff value, custom_sol - your solve_ivp object, custom_ls - your lifespan result

            ######
            Output: matplotlib.pyplot plots, lifespan
        '''

        if (custom_conf is None) or (custom_thr is None) or (custom_sol is None) or (custom_ls is None):
            thr = self.get_threshold()
            K = self.conf[1]
            arr = self.calculate_population()
            life = self.lifespan()
        else:
            thr = custom_thr
            K = custom_conf[1]
            life = custom_ls
            arr = custom_sol

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

        axes[0].plot(self.t*21.5/(365.25*24), Y)
        axes[1].plot(self.t[:-1]*21.5/(365.25*24), deriv)

        axes[0].grid('True')
        axes[1].grid('True')

        axes[1].set_xlabel('Years')
        axes[1].set_ylabel('dM/dt')

        axes[0].set_xlabel('Years')
        axes[0].set_ylabel('M(t)')

        axes[0].set_title('Mortality function')
        axes[1].set_title('Mortality function derivative')

        axes[0].axvline(maximum*21.5/(365.25*24), color = 'r', ls = '--')
        axes[1].axvline(maximum*21.5/(365.25*24), color = 'r', ls = '--')
        axes[1].axhline(max_der, ls = '-.', color = 'g')

        print('Max:', np.round(maximum*21.5/(365.25*24), 0))
        print('Max deriv:', np.round(max_der, 0))
        print('Ratio of max deriv to total lifespan in %:', np.round((maximum*21.5/(365.25*24))/((life*21.5)/(365.25*24))*100, 1))
    
    def variator(
        self, 
        fraction:float = 5, 
        sampling_freq:int = 4, 
        x_bound:float = 300, 
        only_z:bool=False, 
        only_r:bool=False, 
        only_sigma:bool=False, 
        only_alpha:bool=False, 
        z_min:float=0.1, 
        z_max:float=0.9,
        d_min:float=0.1,
        d_max:float=0.9, 
        legend:bool=True,
        custom_conf:List = None,
        custom_thr:float = None,
        custom_init:List = None) -> None:

        '''
            Perturb the parameters of a system to see the results.

            ######
            Args: fraction for interval as {parameter/fraction; parameter*fraction}, sampling_freq - amount of equidistant points to separate the interval, 
            x_bound = cut the plot on this value of time, only_* - variate only * parameter, {z_min; z_max} and {d_min;d_max} - bounds for proportion of alive mutants and their death rate,
            legend - show legend on plot or not, custom_* - define your own parameters and solution for a variator.

            ######
            Output: plots and lifespans
        '''
        
        if (custom_conf is None) or (custom_init is None) or (custom_thr is None):
            init = self.get_initial_conditions()
            thr = self.get_threshold()
            K = self.conf[1]
            config_ = self.conf
        else:
            init = custom_init
            thr = custom_thr
            config_ = custom_conf
            K = config_[1]
        model = self.get_model()
        

        if only_r:
            self._vary_parameter('r', 3, fraction, sampling_freq, model, init, thr*K, x_bound, legend,config_, thr)
        elif only_alpha:
            self._vary_parameter('alpha', 5, fraction, sampling_freq, model, init, thr*K, x_bound, legend, config_, thr)
        elif only_sigma:
            self._vary_parameter('sigma', 0, fraction, sampling_freq, model, init, thr*K, x_bound, legend, config_, thr)
        elif only_z:
            self._vary_z_parameter(z_min, z_max, sampling_freq, model, init, thr*K, x_bound, legend, config_, thr)
        else:
            if self.equation == 'two':
                sols_sigma_ = []
                sigma_range_ = np.linspace(config_[0]/fraction, fraction*config_[0], sampling_freq)       
                sols_alpha_ = []
                alpha_range_ = np.linspace(config_[5]/fraction, fraction*config_[5], sampling_freq)       
                sols_beta_ = []
                beta_range_ = np.linspace(config_[6]/fraction, fraction*config_[6], sampling_freq)        
                sols_epsilon_ = []
                eps_range_ = np.linspace(config_[4]/fraction, fraction*config_[4], sampling_freq)   

                for sigma1 in sigma_range_:
                    conf_sigma = config_.copy()
                    conf_sigma[0] = sigma1
                    solution3_ = solve_ivp(model,t_span=(self.start, self.end), y0=init, t_eval=self.t, method = self.method,args=tuple(conf_sigma), dense_output=True)
                    sols_sigma_.append(solution3_)   

                for a1 in alpha_range_:
                    conf_alpha = config_.copy()
                    conf_alpha[5] = a1
                    solution4_ = solve_ivp(model,t_span=(self.start, self.end), y0=init, t_eval=self.t, method = self.method,args=tuple(conf_alpha), dense_output=True)
                    sols_alpha_.append(solution4_)   

                for b1 in beta_range_:
                    conf_beta = config_.copy()
                    conf_beta[6] = b1
                    solution5_ = solve_ivp(model,t_span=(self.start, self.end), y0=init, t_eval=self.t, method = self.method,args=tuple(conf_beta), dense_output=True)
                    sols_beta_.append(solution5_) 

                for e1 in eps_range_:
                    conf_eps = config_.copy()
                    conf_eps[4] = e1
                    solution6_ = solve_ivp(model,t_span=(self.start, self.end), y0=init, t_eval=self.t, method = self.method,args=tuple(conf_eps), dense_output=True)
                    sols_epsilon_.append(solution6_)  

                fig, ax = plt.subplots(2, 2, figsize = (20,14))  

                for i in range(len(sols_sigma_)):
                    ax[0][0].plot(self.t*21.5/(365.25*24), sols_sigma_[i].y[0], label = f'Somatic population for sigma = {np.round(sigma_range_[i], 4)}')   

                for i in range(len(sols_alpha_)):
                    ax[0][1].plot(self.t*21.5/(365.25*24), sols_alpha_[i].y[0], label = f'Somatic population for alpha = {np.round(alpha_range_[i], 13)}')  

                for i in range(len(sols_beta_)):
                    ax[1][0].plot(self.t*21.5/(365.25*24), sols_beta_[i].y[0], label = f'Somatic population for betas = {np.round(beta_range_[i], 13)}')      

                for i in range(len(sols_epsilon_)):
                    ax[1][1].plot(self.t*21.5/(365.25*24), sols_epsilon_[i].y[0], label = f'Somatic population for epsilon = {np.round(eps_range_[i], 3)}')  

                ax[0][0].grid('True')
                ax[0][1].grid('True')
                ax[1][0].grid('True')
                ax[1][1].grid('True')      

                ax[0][0].set_xlabel('Years')
                ax[0][1].set_xlabel('Years')
                ax[1][0].set_xlabel('Years')
                ax[1][1].set_xlabel('Years')   

                ax[0][0].set_ylabel('Population')
                ax[0][1].set_ylabel('Population')
                ax[1][0].set_ylabel('Population')
                ax[1][1].set_ylabel('Population')   

                ax[0][0].set_title('Population number for different sigmas')
                ax[0][1].set_title('Population number for different somatic mutation rates')
                ax[1][0].set_title('Population number for different stem cell mutation rates')
                ax[1][1].set_title('Population number for different proliferation rates')     

                ax[0][0].set_xlim(0,x_bound)
                ax[0][1].set_xlim(0,x_bound)
                ax[1][0].set_xlim(0,x_bound)
                ax[1][1].set_xlim(0,x_bound)  

                ax[0][0].axhline(thr*K, ls = '--', color = 'r')
                ax[0][1].axhline(thr*K, ls = '--', color = 'r')
                ax[1][0].axhline(thr*K, ls = '--', color = 'r')
                ax[1][1].axhline(thr*K, ls = '--', color = 'r')  

                if legend:
                    ax[0][0].legend(loc = 'upper right')
                    ax[0][1].legend(loc ='upper right')     
                    ax[1][1].legend(loc = 'upper right')
                    ax[1][0].legend(loc = 'upper right')

            elif self.equation == 'single':
                if self.include == True:
                    raise ValueError('Do not use default variator with single-equation without alive mutants. You can vary alpha, sigma, r and z by one.')
                sols_sigma_ = []
                sigma_range_ = np.linspace(config_[0]/fraction, fraction*config_[0], sampling_freq)       
                sols_alpha_ = []
                alpha_range_ = np.linspace(config_[5]/fraction, fraction*config_[5], sampling_freq)       
                sols_d_ = []
                d_range_ = np.linspace(d_min, d_max, sampling_freq)      
                sols_z_ = []
                z_range_ = np.linspace(z_min, z_max, sampling_freq)  

                for sigma1 in sigma_range_:
                    conf_sigma = config_.copy()
                    conf_sigma[0] = sigma1
                    solution3_ = solve_ivp(model,t_span=(self.start, self.end), y0=init, t_eval=self.t, method = self.method,args=tuple(conf_sigma), dense_output=True)
                    sols_sigma_.append(solution3_)  

                for a1 in alpha_range_:
                    conf_alpha = config_.copy()
                    conf_alpha[5] = a1
                    solution4_ = solve_ivp(model,t_span=(self.start, self.end), y0=init, t_eval=self.t, method = self.method,args=tuple(conf_alpha), dense_output=True)
                    sols_alpha_.append(solution4_)   

                for d1 in d_range_:
                    conf_d = config_.copy()
                    conf_d[-1] = d1
                    solution5_ = solve_ivp(model,t_span=(self.start, self.end), y0=init, t_eval=self.t, method = self.method,args=tuple(conf_d), dense_output=True)
                    sols_d_.append(solution5_) 

                for z1 in z_range_:
                    conf_z = config_.copy()
                    conf_z[-2] = z1
                    solution6_ = solve_ivp(model,t_span=(self.start, self.end), y0=init, t_eval=self.t, method = self.method,args=tuple(conf_z), dense_output=True)
                    sols_z_.append(solution6_)  

                fig, ax = plt.subplots(2, 2, figsize = (20,14))   

                for i in range(len(sols_sigma_)):
                    ax[0][0].plot(self.t*21.5/(365.25*24), sols_sigma_[i].y[0], label = f'Somatic population for sigma = {np.round(sigma_range_[i], 4)}')  

                for i in range(len(sols_alpha_)):
                    ax[0][1].plot(self.t*21.5/(365.25*24), sols_alpha_[i].y[0], label = f'Somatic population for alpha = {np.round(alpha_range_[i], 13)}')

                for i in range(len(sols_d_)):
                    ax[1][0].plot(self.t*21.5/(365.25*24), sols_d_[i].y[0], label = f'Somatic population for d = {np.round(d_range_[i], 3)}')  

                for i in range(len(sols_z_)):
                    ax[1][1].plot(self.t*21.5/(365.25*24), sols_z_[i].y[0], label = f'Somatic population for z = {np.round(z_range_[i], 3)}') 

                ax[0][0].grid('True')
                ax[0][1].grid('True')
                ax[1][0].grid('True')
                ax[1][1].grid('True') 

                ax[0][0].set_xlabel('Years')
                ax[0][1].set_xlabel('Years')
                ax[1][0].set_xlabel('Years')
                ax[1][1].set_xlabel('Years')   

                ax[0][0].set_ylabel('Population')
                ax[0][1].set_ylabel('Population')
                ax[1][0].set_ylabel('Population')
                ax[1][1].set_ylabel('Population')    

                ax[0][0].set_title('Population number for different sigmas')
                ax[0][1].set_title('Population number for different somatic mutation rates')
                ax[1][0].set_title('Population number for different alive mutants death rate')
                ax[1][1].set_title('Population number for different proportion of alive mutants')  

                ax[0][0].set_xlim(0,x_bound)
                ax[0][1].set_xlim(0,x_bound)
                ax[1][0].set_xlim(0,x_bound)
                ax[1][1].set_xlim(0,x_bound)    

                ax[0][0].axhline(thr*K, ls = '--', color = 'r')
                ax[0][1].axhline(thr*K, ls = '--', color = 'r')
                ax[1][0].axhline(thr*K, ls = '--', color = 'r')
                ax[1][1].axhline(thr*K, ls = '--', color = 'r')     
                
                if legend:
                    ax[0][0].legend(loc = 'upper right')
                    ax[0][1].legend(loc ='upper right') 
                    ax[1][0].legend(loc = 'upper right')
                    ax[1][1].legend(loc = 'upper right')    


    def _vary_parameter(self, param_name, param_index, fraction, sampling_freq, model, initial_conditions, threshold, x_bound, legend, config,thr):
        param_range = np.linspace(config[param_index] / fraction, fraction * config[param_index], sampling_freq)
        solutions = []

        for param_value in param_range:
            conf = config.copy()
            conf[param_index] = param_value
            solution = solve_ivp(model, t_span=(self.start, self.end), y0=initial_conditions, t_eval=self.t, method=self.method, args=tuple(conf), dense_output=True)
            solutions.append(solution)

        print('Lower bound result:')
        ls_ = self.lifespan(custom_solution=solutions[0], custom_conf=config, custom_thr=thr)
        print('-'*40)
        print('Upper bound result:')
        ls = self.lifespan(custom_solution=solutions[-1],custom_conf=config, custom_thr=thr)

        self._plot_variation(solutions, param_range, param_name, threshold, x_bound, legend)
    
    def _vary_z_parameter(self, z_min, z_max, steps, model, initial_conditions, threshold, x_bound, legend, config, thr):
        if self.equation == 'two':
            raise ValueError('No alive mutants in two equation model. Consider using single equation.')
    
        z_range = np.linspace(z_min, z_max, steps)
        solutions = []
    
        for z_value in z_range:
            conf = config.copy()
            conf[-2] = z_value
            solution = solve_ivp(model, t_span=(self.start, self.end), y0=initial_conditions, t_eval=self.t, method=self.method, args=tuple(conf), dense_output=True)
            solutions.append(solution)

        print('Lower bound result:')
        ls_ = self.lifespan(custom_solution=solutions[0], custom_conf=config, custom_thr=thr)
        print('-'*40)
        print('Upper bound result:')
        ls = self.lifespan(custom_solution=solutions[-1],custom_conf=config, custom_thr=thr)
        self._plot_variation(solutions, z_range, 'alive mutants proportion', threshold, x_bound, legend)
    
    def _plot_variation(self, solutions, param_range, param_name, threshold, x_bound, legend):
        fig, ax = plt.subplots(figsize=(12, 8))

        for solution, param_value in zip(solutions, param_range):
            ax.plot(self.t * 21.5 / (365.25 * 24), solution.y[0], label=f'{param_name} = {np.round(param_value, 13)}')

        self._finalize_plot(ax, param_name, threshold, x_bound, legend)
        
    def _finalize_plot(self, ax, param_name, threshold, x_bound, legend):
        ax.grid(True)
        ax.set_xlabel('Years')
        ax.set_ylabel('Population')
        ax.set_xlim(0, x_bound)
        ax.axhline(threshold, ls='--', color='r')
        if legend:
            ax.legend(loc='upper right')
        plt.show()