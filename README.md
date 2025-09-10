
# Minimal Realization Time-Delay Koopman Analysis for Nonlinear System Identification

**Code for preprint paper** "Minimal Realization Time-Delay Koopman Analysis for Nonlinear System Identification"  

---

### Paper Abstract
Numerical simulations and real-world monitoring data are increasingly prevalent in fields such as solid mechanics, fluid dynamics, and structural engineering. However, 
developing accurate models that capture the underlying system dynamics from sparse and noisy measurements remains a significant challenge. To address this, we propose a 
novel methodology called Minimal Realization Time-Delay Koopman (MRTK) analysis. This method combines time-delay embedding with Koopman operator theory to transform nonlinear 
dynamics into a linearized form. Additionally, it employs Singular Value Decomposition (SVD) to reduce model order, enhancing computational efficiency and accurately 
identifying system dynamics from sparse and noisy measurements. By explicitly modeling noise in the data, we demonstrate that the MRTK method serves as a generalized 
extension of Dynamic Mode Decomposition (DMD), encompassing variants such as Extended DMD and Total Least Squares DMD, while also establishing theoretical connections 
with both HAVOK and Subspace DMD. We validate the proposed approach using simulated 
data from transitional channel flow and the Lorenz system, as well as real-world wind speed measurements from the Hangzhou Bay Bridge. The results show that integrating 
the identified reduced-order model with a Kalman filter enables real-time, accurate estimation and prediction from sparse data. The method achieves high predictive 
accuracy across all scenarios, with the maximum Normalized Mean Squared Error (NMSE) prediction error for the wind speed field being 1.911\%, underscoring its potential 
to advance the prediction and control of complex systems.


### Introduction

 In this work, we propose a novel methodology called **Minimal Realization Time-Delay Koopman (MRTK) Analysis**, designed to identify the minimal degrees of freedom in linear systems. MRTK can handle both full-state and sparse measurements, even in noisy environments, making it highly suitable for complex systems across various domains such as biology, engineering, neuroscience, and epidemiology.

The validation in the paper primarily comes from the fluid dynamics simulation data of transitional channel flow generously shared by Professor Benjamin Herrmann, our own generated Lorenz system data, and real-world temperature and wind speed data from the structural health monitoring system of the Hangzhou Bay Bridge.The specific implementation of the transitional channel flow can be found in the paper ([Data-driven Resolvent Analysis](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/datadriven-resolvent-analysis/0FA58F03E774C7402EA188D3B8F34B0F)).  Due to the monitoring data being available from the government, restrictions apply to the availability of these data under license for the current study, meaning they are not publicly accessible.  


Therefore, as a public MRTK method repository, we only provide the implementation of the MRTK method based on the Lorenz system data here. We believe this is sufficient for users to understand and apply the MRTK algorithm.

---

Since I have already stored the related data and models, each script can be run independently. The `MRTK_Algorithm.py` provided here is an example of applying the MRTK algorithm to Lorenz system data. However, as a user, you can run `MRTK_Algorithm.py` and replace the loaded data with your own dataset to test whether MRTK works for your specific data.

Required libraries:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm
import os
```

If you wish to reproduce the Lorenz system case shown in the MRTK paper, first run `Lorenz_simulation.py` to generate simulated data.  
Required libraries:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
```

Then, based on the simulated data, you can use `MRTK_Algorithm.py` to generate the MRTK model, and finally run `Lorenz_model.py` in combination with the Kalman filter for prediction.  
Required libraries:
```python
import numpy as np
import pandas as pd
np.random.seed(0)
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.signal import welch
```

### Code Description

- **Lorenz_simulation.py**: Generates the Lorenz system data used in the paper and saves it as `Lorenz_data.npy`.
- **MRTK_Algorithm.py**: As an example, constructs the corresponding Minimal Realization Time-Delay Koopman-Analysis model based on the generated Lorenz system data and saves the matrices `Lorenz_A_DMD.npy` and `Lorenz_C.npy`.
- **Lorenz_model.py**: Runs the model with `Lorenz_A_DMD.npy` and `Lorenz_C.npy`, integrating a Kalman filter for prediction.
