# Evaluating impact of somatic mutations on aging: a dynamical system approach
 Dynamical system and calculation of parameters for the publication.
 
![graphical abstract with scheme](https://github.com/Malaevleo/dynamical_system_somatic_mutations_2024/assets/143445560/c9e3a135-596f-4fdf-87ab-970b91727ef3)

# How to use

## First steps

Visual style of plots is defined via the line

```
plt.style.use("bmh")
```

All of the parameters for organs described in the paper are located in the config which is defined using the function below

```
 def set_config(self) -> List:
        configs = {
            'liver': [0.087, 2e11, 2e11/94000, 4/407, 0.064, (3.5e-9)*9.6e-3, (1.83e-9)*9.6e-3, 4/407, 0.9, 0.239],
            'mouse liver': [0.087, 3.37e8, 3.37e8/94000, 63/407, 0.064, 35*(3.5e-9), 35*(1.83e-9), 63/407, 0.9, 0.239],
            'lungs': [0.073, 10.5e9, 0.07*10.5e9, 0.001/407, 0.007, 6.392476819356688e-12, 6.392476819356688e-12 / 1.9126, 0.001/407, 0.9, 0.239],
            'spinal cord': [0.085, 222e6, 0, 0, 0, 0.9047619*(3.5e-9)*0.0013563, 0, 0, 0.9, 0.239]
        }
        return configs.get(self.organ, configs['liver'])
```

As the solving method we use RK45 because after a lot of experimentation we can conclude that it is the most stable one.

## Choosing model

We grant access to the two models featured in the article. 

### Model 1 ('single equation')

This model features somatic cells population and populations of mutated and dead cells. It can be used in cases where there is a small amount of progenitor/stem cells and their impact can be neglected.

It is called 'single equation' because if we remove the feature of mutant and dead cells populations we end up with equation only for somatic population without another equation for stem cells.

This version of the dynamical system can be applied to all of the organs.

Dynamical system:

$\dot X = rX(1 - \frac{X}{K}) - \alpha X - m(t, X)$

$\dot C = rC(1 - \frac{C}{K}) + z \alpha X - \theta C$

$\dot F = \alpha X (1 - z) +  \theta C$

$m(t, X) = \frac{(C+ (1-z)\alpha X t + \theta C t )^{2} \sigma}{2} t^{2} [1 - \frac{X}{K}]$

Code:
```
    def model_one(self, t, y, s, K, M, r, e, a, b, g, z, d) -> List:
        X, C, F = y
        m1 = 0.5 * s * (1 - X / K)
        dXdt = r * X * (1 - X / K) - a * X - m1 * ((C + (1 - z) * a * X * t + d * C * t) ** 2)
        dCdt = r * C * (1 - C / K) + z * a * X - d * C
        dFdt = (1 - z) * a * X + d * C
        return [dXdt, dCdt, dFdt]
```

### Model 2 ('two equations')
This model includes somatic and stem cells populations without accounting for the fact that mutated cells can also divide and make up their own population. 

![scheme-01](https://github.com/Malaevleo/dynamical_system_somatic_mutations_2024/assets/143445560/50f7099b-e087-4358-9dfe-2a4b8b7b806d)

This model can be applied to the liver and lungs and it should always be used in cases where organ has a lot of progenitor/stem cells and their impact can not be neglected. It can't be used with the spinal cord and if you try to do so you will get the error message.

Dynamical system:

$\dot X = rX(1 - \frac{X}{K}) + 2 \epsilon Y - \alpha X - m(t, X, Y)$

$\dot Y = \gamma Y(1 - \frac{Y}{M}) - \beta Y - \epsilon Y$

$m(t, X, Y) = \frac{(\alpha X + \beta Y)^{2} \sigma}{2} t^{2} [1 - \frac{X + Y}{K + M}]$

Code:
```
    def model_two(self, t, y, s, K, M, r, e, a, b, g, z, d) -> List:
        X, Y, m = y
        m1 = 0.5 * s * (1 - (X + Y) / (K + M))
        dXdt = r * X * (1 - X / K) + 2 * e * Y - a * X - m1 * ((a * X + b * Y) ** 2 * t ** 2)
        dYdt = g * Y * (1 - Y / M) - e * Y - b * Y
        dmdt = 0.5 * s * (1 - (X + Y) / (K + M)) * ((a * X + b * Y) ** 2 * t ** 2)
        return [dXdt, dYdt, dmdt]
```

## Obtaining lifespan

To obtain the lifespan of chosen organ do these steps.

Define which organ you want to assess, which model to use, limit the maximum time (measurement units are years) and choose whether you want to include mutant and dead cells population or not (this option can be used only with single equation version).

Example:

```
z = Somatic_LS(organ = 'lungs',equation = 'single', end_time = 300, include_mutants=True)
```

Equation we use in this example is 'single' which correspond to the Model 1. To use Model 2 write equation = 'two'.

To obtain the lifespan value and plot the somatic cells population this code is used:

```
z.plot_curves(view_all = True, plot_thr = True)
```
There are multiple options in the plot_curves function:

view_all - makes plot go up to the end_time value. Otherwise it stops at the time of death.

plot_thr - plots threshold after crossing which organ dies.

proportions - presents values from the y axis be represented not as an amount of cells but as a percent of organ mass (values range from 0 to 1).

population - which population of cells you want to observe. For model 1 you can use 'somatic', 'alive mutants' or 'dead mutants'. For model 2 'somatic' and 'stem'. Also, for model 2 you can print 'mortality function'. By default somatic cells are plotted.

## Varitator

This function allows to observe how changes in different vairables affect overall lifespan of the chosen organ.

```
z.variator(x_bound=300, d_max=0.9, d_min=0.1, fraction = 10, sampling_freq=10, z_min=0.5, z_max=0.9)
```

By deafult only the $\alpha , \theta, z, \sigma$ are varied. If you want you can vary $r$ by using only_r = True option in the variator function.
