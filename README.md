
# Minimal-Realization-Time-Delay-Koopman-Analysis-for-Nonlinear-System-Identification
**Code for preprint Paper** "Minimal Realization Time-Delay Koopman Analysis for Nonlinear System Identification"  
---

#### Paper Abstract
Data is increasingly abundant in fields such as biology, engineering, neuroscience, and epidemiology. However, developing accurate models that capture the dynamics of the underlying system while ensuring interpretability and generalizability remains a significant challenge. To address this, we propose a novel methodology called Minimal Realization Time-Delay Koopman (MRTK) analysis, which is capable of identifying the minimal degrees of freedom in linear systems and handling both full-state and sparse measurements, even in noisy environments. For full-state measurements, we demonstrate that MRTK is equivalent to the Dynamic Mode Decomposition (DMD) method. For sparse measurements, it employs time-delay embedding techniques and the Koopman operator to construct a minimal realization linear model that is diffeomorphic to the attractor of the original system, unveiling the system's physical dynamics from a differential topology perspective. We validate the proposed approach using simulated data from transitional channel flow and the Lorenz system, as well as real-world temperature and wind speed data from the Hangzhou Bay Bridge. Integrating the identified model with a Kalman filter enables accurate estimation and prediction of sparse data. The results demonstrate high predictive accuracy in both scenarios, with the maximum NMSE prediction error for the wind speed field at 1.911%, highlighting the advanced identification capacity of the method and its potential to advance prediction and control of complex systems.

#### Introduction
The paper is available on Research Square at the following link: [Research Square Paper](https://www.researchsquare.com/article/rs-6029043/v1).
In this work, we propose a novel methodology called **Minimal Realization Time-Delay Koopman (MRTK) Analysis**, designed to identify the minimal degrees of freedom in linear systems. MRTK can handle both full-state and sparse measurements, even in noisy environments, making it highly suitable for complex systems across various domains such as biology, engineering, neuroscience, and epidemiology.







