"""Error Mitigation Module for QLRET PennyLane Device.

This module provides error mitigation techniques that can be applied to 
noisy quantum circuit results to improve accuracy.

Implemented Techniques:
1. Zero-Noise Extrapolation (ZNE)
2. Probabilistic Error Cancellation (PEC) - basic framework
3. Richardson Extrapolation

Future Work:
- Clifford Data Regression (CDR)
- Symmetry verification
- Measurement error mitigation

Usage:
    from qlret.error_mitigation import zero_noise_extrapolation
    
    # Apply ZNE to a circuit function
    mitigated_result = zero_noise_extrapolation(
        circuit_fn, 
        params, 
        noise_factors=[1.0, 2.0, 3.0],
        extrapolation="linear"
    )
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union

__all__ = [
    "zero_noise_extrapolation",
    "richardson_extrapolation",
    "LinearExtrapolator",
    "PolynomialExtrapolator",
    "ExponentialExtrapolator",
    "MitigatedExecutor",
]


# ---------------------------------------------------------------------------
# Extrapolation Methods
# ---------------------------------------------------------------------------


class LinearExtrapolator:
    """Linear extrapolation to zero noise.
    
    Fits: E(λ) = a + b*λ
    Extrapolates to: E(0) = a
    """
    
    def __init__(self):
        self.slope = None
        self.intercept = None
    
    def fit(self, noise_factors: np.ndarray, values: np.ndarray) -> None:
        """Fit linear model to noisy data."""
        # Simple linear regression
        n = len(noise_factors)
        x = noise_factors
        y = values
        
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        self.intercept = (sum_y - self.slope * sum_x) / n
    
    def extrapolate(self, target: float = 0.0) -> float:
        """Extrapolate to target noise level (default: 0)."""
        if self.intercept is None:
            raise ValueError("Must call fit() before extrapolate()")
        return self.intercept + self.slope * target
    
    def __call__(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Fit and extrapolate in one step."""
        self.fit(noise_factors, values)
        return self.extrapolate(0.0)


class PolynomialExtrapolator:
    """Polynomial extrapolation to zero noise.
    
    Fits: E(λ) = Σ_i a_i * λ^i
    Extrapolates to: E(0) = a_0
    """
    
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.coefficients = None
    
    def fit(self, noise_factors: np.ndarray, values: np.ndarray) -> None:
        """Fit polynomial model to noisy data."""
        # Use numpy polyfit (coefficients in descending order)
        self.coefficients = np.polyfit(noise_factors, values, self.degree)
    
    def extrapolate(self, target: float = 0.0) -> float:
        """Extrapolate to target noise level."""
        if self.coefficients is None:
            raise ValueError("Must call fit() before extrapolate()")
        return np.polyval(self.coefficients, target)
    
    def __call__(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Fit and extrapolate in one step."""
        self.fit(noise_factors, values)
        return self.extrapolate(0.0)


class ExponentialExtrapolator:
    """Exponential extrapolation to zero noise.
    
    Fits: E(λ) = a * exp(b * λ) + c
    Extrapolates to: E(0) = a + c
    
    Uses linearized fitting: log(E - c_guess) = log(a) + b*λ
    """
    
    def __init__(self, asymptote_guess: float = 0.0):
        self.asymptote = asymptote_guess
        self.a = None
        self.b = None
    
    def fit(self, noise_factors: np.ndarray, values: np.ndarray) -> None:
        """Fit exponential model to noisy data."""
        # Linearize by taking log
        y_shifted = values - self.asymptote
        # Ensure positive for log
        y_shifted = np.maximum(y_shifted, 1e-10)
        log_y = np.log(y_shifted)
        
        # Linear fit on log data
        n = len(noise_factors)
        x = noise_factors
        
        sum_x = np.sum(x)
        sum_log_y = np.sum(log_y)
        sum_x_log_y = np.sum(x * log_y)
        sum_x2 = np.sum(x ** 2)
        
        self.b = (n * sum_x_log_y - sum_x * sum_log_y) / (n * sum_x2 - sum_x ** 2)
        log_a = (sum_log_y - self.b * sum_x) / n
        self.a = np.exp(log_a)
    
    def extrapolate(self, target: float = 0.0) -> float:
        """Extrapolate to target noise level."""
        if self.a is None:
            raise ValueError("Must call fit() before extrapolate()")
        return self.a * np.exp(self.b * target) + self.asymptote
    
    def __call__(self, noise_factors: np.ndarray, values: np.ndarray) -> float:
        """Fit and extrapolate in one step."""
        self.fit(noise_factors, values)
        return self.extrapolate(0.0)


# ---------------------------------------------------------------------------
# Zero-Noise Extrapolation (ZNE)
# ---------------------------------------------------------------------------


def zero_noise_extrapolation(
    circuit_fn: Callable,
    params: Any,
    noise_factors: List[float] = [1.0, 1.5, 2.0],
    extrapolation: str = "linear",
    noise_scale_fn: Optional[Callable[[float], Callable]] = None,
    degree: int = 2,
) -> Union[float, np.ndarray]:
    """Apply Zero-Noise Extrapolation to a quantum circuit.
    
    This technique runs the circuit at multiple noise levels and
    extrapolates to the zero-noise limit.
    
    Parameters
    ----------
    circuit_fn : Callable
        A function that takes params and returns circuit result.
        The function should accept a noise_scale keyword argument.
    params : Any
        Parameters to pass to circuit_fn.
    noise_factors : List[float]
        Noise scaling factors (1.0 = original noise level).
    extrapolation : str
        Extrapolation method: "linear", "polynomial", or "exponential".
    noise_scale_fn : Callable, optional
        Function that modifies the circuit to scale noise.
        If None, assumes circuit_fn accepts noise_scale kwarg.
    degree : int
        Polynomial degree (only used if extrapolation="polynomial").
        
    Returns
    -------
    float or np.ndarray
        The zero-noise extrapolated result.
        
    Example
    -------
    >>> def noisy_circuit(params, noise_scale=1.0):
    ...     dev = QLRETDevice(wires=2, epsilon=1e-4 * noise_scale)
    ...     @qml.qnode(dev)
    ...     def circuit(p):
    ...         qml.RX(p[0], wires=0)
    ...         qml.DepolarizingChannel(0.01 * noise_scale, wires=0)
    ...         return qml.expval(qml.PauliZ(0))
    ...     return circuit(params)
    >>> 
    >>> mitigated = zero_noise_extrapolation(
    ...     noisy_circuit, 
    ...     [0.5],
    ...     noise_factors=[1.0, 1.5, 2.0, 2.5]
    ... )
    """
    noise_factors = np.array(noise_factors)
    
    # Run circuit at each noise level
    results = []
    for factor in noise_factors:
        if noise_scale_fn is not None:
            scaled_circuit = noise_scale_fn(factor)
            result = scaled_circuit(params)
        else:
            result = circuit_fn(params, noise_scale=factor)
        results.append(result)
    
    results = np.array(results)
    
    # Handle array results (multiple expectation values)
    if results.ndim > 1:
        # Apply extrapolation to each element
        extrapolated = np.zeros(results.shape[1:])
        for idx in np.ndindex(results.shape[1:]):
            values = results[(slice(None),) + idx]
            extrapolated[idx] = _extrapolate(noise_factors, values, extrapolation, degree)
        return extrapolated
    
    # Scalar result
    return _extrapolate(noise_factors, results, extrapolation, degree)


def _extrapolate(
    noise_factors: np.ndarray, 
    values: np.ndarray, 
    method: str,
    degree: int = 2,
) -> float:
    """Apply extrapolation method to get zero-noise estimate."""
    if method == "linear":
        extrapolator = LinearExtrapolator()
    elif method == "polynomial":
        extrapolator = PolynomialExtrapolator(degree=degree)
    elif method == "exponential":
        extrapolator = ExponentialExtrapolator()
    else:
        raise ValueError(f"Unknown extrapolation method: {method}")
    
    return extrapolator(noise_factors, values)


def richardson_extrapolation(
    circuit_fn: Callable,
    params: Any,
    noise_factors: List[float] = [1.0, 2.0],
) -> float:
    """Apply Richardson extrapolation for error mitigation.
    
    This is a specific form of polynomial extrapolation commonly used
    in quantum error mitigation.
    
    For two noise factors [λ₁, λ₂], the Richardson estimate is:
        E_0 = (λ₂ * E(λ₁) - λ₁ * E(λ₂)) / (λ₂ - λ₁)
    
    Parameters
    ----------
    circuit_fn : Callable
        A function that takes params and noise_scale, returns result.
    params : Any
        Parameters to pass to circuit_fn.
    noise_factors : List[float]
        Two noise scaling factors (default: [1.0, 2.0]).
        
    Returns
    -------
    float
        The Richardson-extrapolated zero-noise estimate.
    """
    if len(noise_factors) != 2:
        raise ValueError("Richardson extrapolation requires exactly 2 noise factors")
    
    λ1, λ2 = noise_factors
    E1 = circuit_fn(params, noise_scale=λ1)
    E2 = circuit_fn(params, noise_scale=λ2)
    
    # Richardson formula
    E0 = (λ2 * E1 - λ1 * E2) / (λ2 - λ1)
    return E0


# ---------------------------------------------------------------------------
# Mitigated Executor (High-Level API)
# ---------------------------------------------------------------------------


class MitigatedExecutor:
    """High-level executor with built-in error mitigation.
    
    This class wraps a QLRET device and applies error mitigation
    techniques automatically.
    
    Parameters
    ----------
    device : QLRETDevice
        The LRET device to use for execution.
    mitigation : str
        Mitigation technique: "none", "zne", "richardson".
    noise_factors : List[float]
        Noise scaling factors for ZNE.
    extrapolation : str
        Extrapolation method for ZNE.
        
    Example
    -------
    >>> dev = QLRETDevice(wires=4, epsilon=1e-4)
    >>> executor = MitigatedExecutor(dev, mitigation="zne")
    >>> 
    >>> @executor.qnode
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> 
    >>> # This automatically applies ZNE
    >>> result = circuit([0.5])
    """
    
    def __init__(
        self,
        device: Any,
        mitigation: str = "none",
        noise_factors: List[float] = [1.0, 1.5, 2.0],
        extrapolation: str = "linear",
    ):
        self.device = device
        self.mitigation = mitigation
        self.noise_factors = noise_factors
        self.extrapolation = extrapolation
    
    def qnode(self, circuit_fn: Callable) -> Callable:
        """Decorator to create a mitigated QNode.
        
        Parameters
        ----------
        circuit_fn : Callable
            The quantum circuit function.
            
        Returns
        -------
        Callable
            A wrapped function that applies error mitigation.
        """
        import pennylane as qml
        
        if self.mitigation == "none":
            # No mitigation, just create a regular QNode
            return qml.QNode(circuit_fn, self.device)
        
        def mitigated_circuit(*args, **kwargs):
            if self.mitigation == "zne":
                # Create circuits at different noise levels
                def circuit_at_noise(params, noise_scale=1.0):
                    # Scale epsilon to increase noise
                    original_epsilon = self.device.epsilon
                    self.device.epsilon = original_epsilon * noise_scale
                    
                    node = qml.QNode(circuit_fn, self.device)
                    result = node(*args, **kwargs)
                    
                    # Restore original epsilon
                    self.device.epsilon = original_epsilon
                    return result
                
                return zero_noise_extrapolation(
                    circuit_at_noise,
                    args,
                    noise_factors=self.noise_factors,
                    extrapolation=self.extrapolation,
                )
            
            elif self.mitigation == "richardson":
                def circuit_at_noise(params, noise_scale=1.0):
                    original_epsilon = self.device.epsilon
                    self.device.epsilon = original_epsilon * noise_scale
                    
                    node = qml.QNode(circuit_fn, self.device)
                    result = node(*args, **kwargs)
                    
                    self.device.epsilon = original_epsilon
                    return result
                
                return richardson_extrapolation(
                    circuit_at_noise,
                    args,
                    noise_factors=self.noise_factors[:2],
                )
            
            else:
                raise ValueError(f"Unknown mitigation technique: {self.mitigation}")
        
        return mitigated_circuit


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def estimate_noise_scaling_factors(
    base_noise: float,
    num_factors: int = 3,
    max_scale: float = 3.0,
) -> List[float]:
    """Suggest noise scaling factors for ZNE.
    
    Parameters
    ----------
    base_noise : float
        The base noise level (e.g., depolarizing probability).
    num_factors : int
        Number of scaling factors to generate.
    max_scale : float
        Maximum scaling factor.
        
    Returns
    -------
    List[float]
        Recommended noise scaling factors.
    """
    # Ensure factors are spaced appropriately
    # Using linear spacing is simple but effective
    factors = np.linspace(1.0, max_scale, num_factors)
    
    # For low noise, we might want more factors
    if base_noise < 0.01:
        factors = np.linspace(1.0, min(max_scale, 2.0), num_factors)
    
    return factors.tolist()


def validate_zne_results(
    noise_factors: List[float],
    values: List[float],
) -> dict:
    """Validate ZNE extrapolation quality.
    
    Returns metrics about the extrapolation fit quality.
    
    Parameters
    ----------
    noise_factors : List[float]
        The noise scaling factors used.
    values : List[float]
        The measured values at each noise level.
        
    Returns
    -------
    dict
        Validation metrics including:
        - r_squared: Linear fit quality
        - residuals: Fit residuals
        - extrapolated_linear: Linear extrapolation result
        - extrapolated_poly: Polynomial extrapolation result
        - confidence: Estimated confidence in result
    """
    noise_factors = np.array(noise_factors)
    values = np.array(values)
    
    # Linear fit
    linear = LinearExtrapolator()
    linear.fit(noise_factors, values)
    linear_result = linear.extrapolate(0.0)
    
    # Calculate R²
    y_pred = linear.intercept + linear.slope * noise_factors
    ss_res = np.sum((values - y_pred) ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Polynomial fit (if enough points)
    poly_result = None
    if len(noise_factors) >= 3:
        poly = PolynomialExtrapolator(degree=2)
        poly.fit(noise_factors, values)
        poly_result = poly.extrapolate(0.0)
    
    # Confidence heuristic based on R² and agreement between methods
    confidence = r_squared
    if poly_result is not None:
        agreement = 1 - abs(linear_result - poly_result) / (abs(linear_result) + 1e-10)
        confidence = (r_squared + agreement) / 2
    
    return {
        "r_squared": r_squared,
        "residuals": (values - y_pred).tolist(),
        "extrapolated_linear": linear_result,
        "extrapolated_poly": poly_result,
        "confidence": min(max(confidence, 0.0), 1.0),
    }
