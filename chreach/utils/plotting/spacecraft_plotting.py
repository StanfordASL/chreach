"""Helper functions for plotting trajectories of spacecraft system."""
from typing import Dict, Any
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc, rcParams
from chreach.reach import SampledReachableSetTrajectory


def plot_optimal_control_solution(
    discretization_time: float,
    horizon: int,
    constraints_parameters: Dict[str, Any],
    states: jnp.ndarray, # (horizon+1, num_states)
    controls: jnp.ndarray, # (horizon, num_controls)
    reference_state: jnp.ndarray): # (num_states)
    times = discretization_time * jnp.arange(horizon + 1)

    fig = plt.figure(figsize=[6,3])
    colors = ['r', 'g', 'b', 'm']
    # q-trajectory
    plt.plot(times, states[:, 0], color=colors[0], label=r'$q_1$')
    plt.plot(times, states[:, 1], color=colors[1], label=r'$q_2$')
    plt.plot(times, states[:, 2], color=colors[2], label=r'$q_3$')
    plt.plot(times, states[:, 3], color=colors[3], label=r'$q_4$')
    for i in range(4):
        # reference
        plt.plot(
            times,
            jnp.ones(horizon + 1) * reference_state[i],
            'k--')
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$q(t)$', fontsize=24, rotation=0, labelpad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim([-0.55, 1.05])
    plt.legend(loc='center',
        fontsize=20, handletextpad=0.5, 
        labelspacing=0.2, handlelength=1)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()

    fig = plt.figure(figsize=[6,3])
    for i in range(3):
        # omega-trajectory
        plt.plot(
            times,
            states[:, 4+i],
            label=str(i), color=colors[i])
        # reference
        plt.plot(
            times,
            jnp.ones(horizon + 1) * reference_state[4+i],
            'k--')
    plt.plot(
        times,
        -constraints_parameters["max_state"] * jnp.ones(horizon + 1),
        'r--')
    plt.plot(
        times,
        constraints_parameters["max_state"] * jnp.ones(horizon + 1),
        'r--')
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$\omega(t)$', fontsize=24, rotation=0, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()

    fig = plt.figure(figsize=[6, 3])
    for i in range(3):
        plt.plot(
            times[:-1],
            controls[:, i],
            color=colors[i], alpha=0.05)
    plt.plot(
        times[:-1],
        -constraints_parameters["max_control"] * jnp.ones(horizon),
        'r--')
    plt.plot(
        times[:-1],
        constraints_parameters["max_control"] * jnp.ones(horizon),
        'r--')
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$u(t)$', fontsize=24, rotation=0, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()
    plt.show()


def plot_mpc_solutions(
    num_mpc_runs: int,
    discretization_time: float,
    horizon: int,
    constraints_parameters: Dict[str, Any],
    states: jnp.ndarray, # (num_mpc_runs, horizon+1, num_states)
    controls: jnp.ndarray, # (num_mpc_runs, horizon, num_controls)
    reference_state: jnp.ndarray): # (num_mpc_runs)
    times = discretization_time * jnp.arange(horizon + 1)

    fig = plt.figure(figsize=[6, 4])
    colors = ['r', 'g', 'b', 'm']
    # q-trajectory
    for run in range(num_mpc_runs):
        plt.plot(times, states[run, :, 0], color=colors[0], alpha=0.1)
        plt.plot(times, states[run, :, 1], color=colors[1], alpha=0.1)
        plt.plot(times, states[run, :, 2], color=colors[2], alpha=0.1)
        plt.plot(times, states[run, :, 3], color=colors[3], alpha=0.1)
    for i in range(4):
        # reference
        plt.plot(
            times,
            jnp.ones(horizon + 1) * reference_state[i],
            'k--')
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$q(t)$', fontsize=24, rotation=0, labelpad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim([-1.05, 1.05])
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()

    fig = plt.figure(figsize=[6, 4])
    for i in range(3):
        # omega-trajectory
        for run in range(num_mpc_runs):
            plt.plot(
                times,
                states[run, :, 4+i],
                label=str(i), color=colors[i], alpha=0.1)
        # reference
        plt.plot(
            times,
            jnp.ones(horizon + 1) * reference_state[4+i],
            'k--')
    plt.plot(
        times,
        -constraints_parameters["max_state"] * jnp.ones(horizon + 1),
        'r--')
    plt.plot(
        times,
        constraints_parameters["max_state"] * jnp.ones(horizon + 1),
        'r--')
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$\omega(t)$', fontsize=24, rotation=0, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()

    fig = plt.figure(figsize=[6, 4])
    plt.plot(
        times[:-1],
        -constraints_parameters["max_control"] * jnp.ones(horizon),
        'r--')
    plt.plot(
        times[:-1],
        constraints_parameters["max_control"] * jnp.ones(horizon),
        'r--')
    for i in range(3):
        for run in range(num_mpc_runs):
            plt.plot(
                times[:-1],
                controls[run, :, i],
                color=colors[i], alpha=0.1)
    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$u(t)$', fontsize=24, rotation=0, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(which='minor', alpha=0.5, linestyle='--')
    plt.grid(which='major', alpha=0.75, linestyle=':')
    plt.tight_layout()
    plt.show()


def plot_computation_times(
    num_scp_iters_max: int,
    sample_sizes_for_computation_times: int,
    solver_parameters: Dict[str, Any],
    computation_time_per_iteration_define: jnp.ndarray,
    computation_time_per_iteration_solve: jnp.ndarray,
    computation_times_cumulative: jnp.ndarray,
    accuracy_errors: jnp.ndarray):
    num_scp_iters_max = solver_parameters['num_scp_iterations']
    computation_times_define = computation_time_per_iteration_define[:, :, :num_scp_iters_max]
    computation_times_solve = computation_time_per_iteration_solve[:, :, :num_scp_iters_max]
    computation_times_cum = computation_times_cumulative[:, :, :num_scp_iters_max]
    accuracy_error = accuracy_errors[:, :, :num_scp_iters_max]
    sample_sizes_for_computation_times = sample_sizes_for_computation_times

    # scp errors
    idx = 1
    first_scp_iter = 0
    accuracy_error_median = np.median(accuracy_error, axis=0)
    accuracy_error_median = accuracy_error_median[idx, :]
    accuracy_error_median = accuracy_error_median[first_scp_iter:]
    scp_iters = np.arange(num_scp_iters_max)[first_scp_iter:] + 1
    print("plotting for M =", sample_sizes_for_computation_times[idx])
    print("plotting for scp_iters =", scp_iters)
    fig = plt.figure(figsize=[10, 4])
    plt.grid()
    plt.scatter(
        scp_iters,
        accuracy_error_median,
        color='k')
    plt.plot(
        scp_iters,
        accuracy_error_median,
        color='k')
    plt.yscale('log') 
    plt.xlabel(r'SCP Iteration $k$', fontsize=24)
    plt.ylabel(r'Relative error', fontsize=24)
    plt.ylabel(r'$\frac{\|u^{k}-u^{k-1}\|}{\|u^{k}\|}$', 
        fontsize=32, rotation=0, labelpad=60)
    plt.xticks([i for i in scp_iters], ([i for i in scp_iters]))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    # computation times as a function of scp iter
    width = 0.35 
    fig = plt.subplots(figsize = (10, 4))
    plt.grid(axis='y')
    computation_times_define = np.median(computation_times_define, axis=0)
    computation_times_solve = np.median(computation_times_solve, axis=0)
    computation_times_cum = np.median(computation_times_cum, axis=0)
    computation_times_define = computation_times_define[idx, :]
    computation_times_solve = computation_times_solve[idx, :]
    computation_times_cum = computation_times_cum[idx, :]
    computation_times_define = computation_times_define[first_scp_iter:]
    computation_times_solve = computation_times_solve[first_scp_iter:]
    computation_times_cum = computation_times_cum[first_scp_iter:]
    p1 = plt.bar(scp_iters, 
        1e3 * computation_times_define, 
        width,
        yerr =  0 * scp_iters,
        color='#0C7BDC')
    p2 = plt.bar(scp_iters, 
        1e3 * computation_times_solve, 
        width, 
        bottom = 1e3 * computation_times_define,
        yerr =  0 * scp_iters,
        color='#FFC20A')
    plt.legend((p1[0], p2[0]), ('define', 'solve'),
        fontsize=20,
        loc='upper center')
    plt.xlabel(r'SCP iteration $k$', fontsize=24)
    plt.ylabel(r'Time / SCP iter. (ms)', fontsize=24)
    plt.xlim([np.min(scp_iters)-1, np.max(scp_iters)+1])
    plt.xticks([i for i in scp_iters], ([i for i in scp_iters]))
    plt.xticks([i for i in scp_iters], ([i for i in scp_iters]))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax2 = plt.gca().twinx()
    ax2.plot(
        scp_iters, 
        1e3 * computation_times_cum, 
        'k--',
        linewidth=3)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_ylabel(r'Total time (ms)', fontsize=24)
    plt.tight_layout()

    # computation times as a function of M
    num_scp_iters_max = solver_parameters['num_scp_iterations']
    computation_times_define = \
        computation_time_per_iteration_define[:, :, :num_scp_iters_max]
    computation_times_solve = \
        computation_time_per_iteration_solve[:, :, :num_scp_iters_max]
    computation_times_cum = \
        computation_times_cumulative[:, :, :num_scp_iters_max]
    accuracy_error = \
        accuracy_errors[:, :, :num_scp_iters_max]

    scp_iter = 10 - 1 # python indexing -> this gives the 10th scp iteration
    computation_times_cum = np.median(computation_times_cum, axis=0)
    computation_times_cum_all_sample_sizes_for_computation_times = \
        computation_times_cum[:, scp_iter]
    fig = plt.subplots(figsize = (6, 3))
    plt.grid(axis='y')
    sample_sizes_for_computation_times_vec = np.array(
        [i for i in range(len(sample_sizes_for_computation_times))])
    plt.bar(sample_sizes_for_computation_times_vec, 
        1e3 * computation_times_cum_all_sample_sizes_for_computation_times, 
        2*width,
        yerr =  0 * np.array(sample_sizes_for_computation_times),
        color='#0C7BDC')
    plt.xlabel(r'Sample size $M$', fontsize=24)
    plt.ylabel(r'Time (ms)', fontsize=24)
    plt.xticks(sample_sizes_for_computation_times_vec,
        ([i for i in sample_sizes_for_computation_times]))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()



def plot_reachable_sets(
    discretization_time: float,
    horizon: int,
    constraints_parameters: Dict[str, Any],
    reach_sets_extremal: SampledReachableSetTrajectory,
    error_bounds: jnp.ndarray, # (horizon+1)
    ellipsoids_centers: jnp.ndarray, # (horizon+1, num_states, num_states)
    ellipsoids_shape_matrices: jnp.ndarray, # (horizon+1, num_states)
    reach_sets_randup: SampledReachableSetTrajectory):
    states_extremal = reach_sets_extremal.state_trajectories
    states_randup = reach_sets_randup.state_trajectories


    times = discretization_time * jnp.arange(horizon + 1)
    for j in range(1, 3):
        fig = plt.figure(figsize=[5.25, 5])
        # Lipschitz-based tube
        ys_min = ellipsoids_centers[:, j] - np.sqrt(
            ellipsoids_shape_matrices[:, j, j])
        ys_max = ellipsoids_centers[:, j] + np.sqrt(
            ellipsoids_shape_matrices[:, j, j])
        plt.fill_between(
            times, ys_min, ys_max, 
            facecolor=(0, 0, 1, 0.15), edgecolor=(0, 0, 1, 0.5),
            linestyle='dashed', linewidth=2,
            label=r'$\textrm{Lipschitz tube}$')
        # extremal trajectories approach
        ys_min = jnp.min(states_extremal[:, :, j], axis=0)
        ys_max = jnp.max(states_extremal[:, :, j], axis=0)
        plt.fill_between(
            times, ys_min-error_bounds, ys_max+error_bounds, 
            facecolor=(0, 0, 0, 0.0), edgecolor='k',
            linestyle='dashed', linewidth=2,
            label=r'$\textrm{Algorithm 1}\oplus\epsilon_t$')
        plt.fill_between(
            times, ys_min, ys_max, 
            facecolor=(0 ,0, 1, 0.), edgecolor=(0, 0, 1, 1.),
            linewidth=2,
            label=r'$\textrm{Algorithm 1}$')
        # naive sampling approach
        ys_min = jnp.min(states_randup[:, :, j], axis=0)
        ys_max = jnp.max(states_randup[:, :, j], axis=0)
        plt.fill_between(times, ys_min, ys_max, 
            facecolor=(0 ,0, 1, 0.), edgecolor=(0, 0, 1, 1.))
        plt.fill_between(times, ys_min, ys_max, 
            alpha=0.4, color='b', hatch="X",
            edgecolor='b', linewidth=1,
            label=r'$\textrm{RandUP}$')
        # constraints
        plt.plot(
            times, -constraints_parameters["max_state"] * jnp.ones(horizon + 1),
            'r--', linewidth=2)
        plt.plot(
            times, constraints_parameters["max_state"] * jnp.ones(horizon + 1),
            'r--', linewidth=2)
        plt.xlabel(r'$t$', fontsize=24)
        plt.title(
            r'\textrm{Reachable set estimates for }$\omega_'+str(int(1+j))+'(t)$', 
            fontsize=24, pad=10)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(which='minor', alpha=0.5, linestyle='--')
        plt.grid(which='major', alpha=0.75, linestyle=':')
        if j == 1:
            plt.legend(fontsize=20, loc='lower right', framealpha=0.9)
            plt.ylim((-1.04*constraints_parameters["max_state"], 0.131))
            plt.text(
                2.0, 1.1*constraints_parameters["max_state"],
                r'$\omega_{\textrm{max}}$', 
                fontsize=24, color='r')
        if j == 2:
            # plt.legend(fontsize=18, loc='lower left')
            plt.ylim((0.047, 0.12))
            plt.text(
                6.5, 1.05*constraints_parameters["max_state"],
                r'$\omega_{\textrm{max}}$', 
                fontsize=24, color='r')
        plt.tight_layout()
    plt.show()
