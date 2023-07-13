using Random
using LinearAlgebra 
using Statistics
using Distributions
using JLD

include("integrators.jl")
include("nonconvex_2d.jl")
data_dir = "../../libre/darshans/data/"
state_dir = "../../libre/darshans/states/"
params_path(L, batch, seed, eta, beta, burn_in_time, T, dt, sampling_interval, runs) = string("_L", L, "_batch", 
    batch, "_seed", seed, "_eta", eta, "_beta", beta, "_burn", burn_in_time, "_T", T, "_dt", dt, 
    "_interval", sampling_interval, "_numruns", runs, ".jld")


F(x) = [sin(x[2]), 0]


function coupling_dist(x, y)
    return norm(x - y)
end

output_dim = 1

function main(coupling, L, eta, beta, burn_in_time, T, dt, sampling_interval, num_runs, batchs, seed)
    #println(saves)
    X_0 = [1., 0]

    U, grad_U = get_potential(L)

    function drift(x, eta)
        return -grad_U(x) + eta*F(x)
    end

    Random.seed!(seed)
    RNG = copy(Random.default_rng())

    N = floor(Int64, T/dt)

    runs_per_batch = num_runs ÷ batchs

    Dist_traj = zeros(runs_per_batch, N ÷ sampling_interval)
    Dist_avg = zeros(runs_per_batch, N ÷ sampling_interval)


    if coupling == "indep"
        for batch in 1:batchs
            states = ()
            for j in 1:runs_per_batch
                Dist_traj[j, :], Dist_avg[j, :], state = compute_response(X_0, drift, indep_noise, 
                    coupling_dist, output_dim, eta, beta, burn_in_time, T, dt, sampling_interval, 
                    RNG)
                states = (states..., state)
            end
            save(data_dir*"indep_nonconvex2D_dist"*params_path(L, batch, seed, eta, beta, burn_in_time, 
                T, dt, sampling_interval, runs_per_batch), "Dist_traj", Dist_traj, "Dist_avg", 
                Dist_avg)
            save(state_dir*"indep_nonconvex2D_dist_state"*params_path(L, batch, seed, eta, beta, 
                burn_in_time, T, dt, sampling_interval, runs_per_batch), "states", states)
        end
    elseif coupling == "sync"
        for batch in 1:batchs
            states = ()
            for j in 1:runs_per_batch
                Dist_traj[j, :], Dist_avg[j, :], state = compute_response(X_0, drift, sync_noise, 
                    coupling_dist, output_dim, eta, beta, burn_in_time, T, dt, sampling_interval, 
                    RNG)
                states = (states..., state)
            end
            save(data_dir*"sync_nonconvex2D_dist"*params_path(L, batch, seed, eta, beta, burn_in_time, 
                T, dt, sampling_interval, runs_per_batch), "Dist_traj", Dist_traj, "Dist_avg", 
                Dist_avg)
            save(state_dir*"sync_nonconvex2D_dist_state"*params_path(L, batch, seed, eta, beta, 
                burn_in_time, T, dt, sampling_interval, runs_per_batch), "states", states)
        end
    elseif coupling == "reflect"
        for batch in 1:batchs
            states = ()
            for j in 1:runs_per_batch
                Dist_traj[j, :], Dist_avg[j, :], state = compute_response(X_0, drift, reflect_noise, 
                    coupling_dist, output_dim, eta, beta, burn_in_time, T, dt, sampling_interval, 
                    RNG)
                states = (states..., state)
            end
            save(data_dir*"reflect_nonconvex2D_dist"*params_path(L, batch, seed, eta, beta, burn_in_time, 
                T, dt, sampling_interval, runs_per_batch), "Dist_traj", Dist_traj, "Dist_avg", 
                Dist_avg)
            save(state_dir*"reflect_nonconvex2D_dist_state"*params_path(L, batch, seed, eta, beta, 
                burn_in_time, T, dt, sampling_interval, runs_per_batch), "states", states)
        end
    elseif coupling == "sticky"
        for batch in 1:batchs
            states = ()
            for j in 1:runs_per_batch
                Dist_traj[j, :], Dist_avg[j, :], state = compute_response(X_0, drift, sticky_noise, 
                    coupling_dist, output_dim, eta, beta, burn_in_time, T, dt, sampling_interval, 
                    RNG)
                states = (states..., state)
            end
            save(data_dir*"sticky_nonconvex2D_traj"*params_path(L, batch, seed, eta, beta, burn_in_time, 
                T, dt, sampling_interval, runs_per_batch), "Dist_traj", Dist_traj, "Dist_avg", 
                Dist_avg)
            save(state_dir*"sticky_nonconvex2D_dist_state"*params_path(L, batch, seed, eta, beta, 
                burn_in_time, T, dt, sampling_interval, runs_per_batch), "states", states)
        end
    else
        throw(ArgumentError(string(coupling, "not a valid coupling type argument", 
            "\nValid coupling type arguments: \"indep\", \"sync\", \"reflect\", \"sticky\"")))
    end

end


if abspath(PROGRAM_FILE) == @__FILE__
    coupling = ARGS[1]
    L = parse(Float64, ARGS[2])
    eta = parse(Float64, ARGS[3])
    beta = parse(Float64, ARGS[4])
    burn = parse(Float64, ARGS[5])
    T = parse(Float64, ARGS[6])
    dt = parse(Float64, ARGS[7])
    interval = parse(Int64, ARGS[8])
    num_runs = parse(Int64, ARGS[9])
    batchs = parse(Int64, ARGS[10])
    seed = parse(Int64, ARGS[11])
    

    print(string("Started: Two dimensional Non-Convex coupling distance with ", coupling, " coupling, L = ", L, "\n",
        "eta = ", eta, ", beta = ", beta, ", burn = ", burn, ", T = ", T, ", dt = ", dt, ",\n",  
        "sampling interval = ", interval, ", batchs = ", batchs, ", total runs = ", num_runs, 
        ", seed = ", seed, "\n\n"))

    main(coupling, L, eta, beta, burn, T, dt, interval, num_runs, batchs, seed)

    print(string("Finished: Two dimensional Non-Convex coupling distance with ", coupling, " coupling, L = ", L, "\n",
        "eta = ", eta, ", beta = ", beta, ", burn = ", burn, ", T = ", T, ", dt = ", dt, ",\n",  
        "sampling interval = ", interval, ", batchs = ", batchs, ", total runs = ", num_runs, 
        ", seed = ", seed, "\n\n"))
end