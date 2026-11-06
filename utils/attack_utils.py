import torch
import numpy as np
from typing import Dict, Union, Tuple, Callable




def expansion_phase(
    evaluate_func: Callable[[float], float],
    start: float,
    delta: float,
    ratio: float,
    evals: int
) -> Tuple[float, float, float, int]:
    """
    Generic expansion phase for line search optimization.
    
    Aggressively explores the search space by increasing the step size (delta) when
    a better value is found.
    
    Parameters
    ----------
    evaluate_func : Callable[[float], float]
        Function that takes current_tau and returns evaluation score (higher = better)
    start : float
        Initial tau value
    delta : float
        Initial step size
    ratio : float
        Contraction ratio (used when stopping expansion)
    evals : int
        Maximum number of evaluations
    
    Returns
    -------
    tuple
        (best_tau, best_distance, delta, remaining_evals)
        - best_tau: Best tau value found
        - best_distance: Best evaluation score found
        - delta: Final step size (may be modified)
        - remaining_evals: Number of evaluations remaining
    """
    best_x = start
    best_y = evaluate_func(best_x)
    current_delta = delta
    remaining_evals = evals - 1
    
    while remaining_evals > 0:
        prop_x = best_x + current_delta
        prop_y = evaluate_func(prop_x)
        remaining_evals -= 1
        
        if prop_y > best_y:
            # If the new value is better: Update best_x, double the step size, and continue
            best_x, best_y = prop_x, prop_y
            current_delta *= 2
        else:
            # If the new value is worse: Stop expansion and proceed to contraction
            current_delta *= ratio
            break
    
    return best_x, best_y, current_delta, remaining_evals


def contraction_phase(
    evaluate_func: Callable[[float], float],
    best_x: float,
    best_y: float,
    delta: float,
    ratio: float,
    remaining_evals: int
) -> float:
    """
    Generic contraction phase for line search optimization.
    
    Refines the search by reducing the step size (delta) and searching around the
    current best value.
    
    Parameters
    ----------
    evaluate_func : Callable[[float], float]
        Function that takes current_tau and returns evaluation score (higher = better)
    best_x : float
        Current best tau value
    best_y : float
        Current best distance value
    delta : float
        Current step size
    ratio : float
        Contraction ratio used to reduce step size
    remaining_evals : int
        Remaining evaluations
    
    Returns
    -------
    float
        Optimal tau value found
    """
    current_delta = delta
    current_best_x = best_x
    current_best_y = best_y
    remaining = remaining_evals
    
    while remaining > 0:
        prop_x = current_best_x + current_delta
        prop_y = evaluate_func(prop_x)
        remaining -= 1
        
        if prop_y > current_best_y:
            # If a better value is found: Update best_x and best_y
            current_best_x, current_best_y = prop_x, prop_y
        
        current_delta *= ratio
    
    return current_best_x


def line_search_optimize(
    evaluate_func: Callable[[float], float],
    start: float,
    delta: float,
    ratio: float,
    evals: int
) -> float:
    """
    Perform full line search optimization (expansion + contraction phases).
    
    This function combines expansion_phase() and contraction_phase() into a single
    call for convenience, eliminating the need for attack classes to manage
    the two-phase process manually.
    
    Parameters
    ----------
    evaluate_func : Callable[[float], float]
        Function that takes a factor/tau value and returns evaluation score (higher = better)
    start : float
        Initial value to evaluate
    delta : float
        Initial step size
    ratio : float
        Contraction ratio used to reduce step size
    evals : int
        Maximum number of evaluations
    
    Returns
    -------
    float
        Optimal factor/tau value found
    """
    # Expansion phase
    best_x, best_y, final_delta, remaining_evals = expansion_phase(
        evaluate_func, start, delta, ratio, evals
    )
    
    # Contraction phase
    optimal_value = contraction_phase(
        evaluate_func, best_x, best_y, final_delta, ratio, remaining_evals
    )
    
    return optimal_value