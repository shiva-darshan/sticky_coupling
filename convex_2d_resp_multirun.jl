using Random
using LinearAlgebra 
using Statistics
using Distributions
using JLD

include("integrators.jl")
data_dir = "../../libre/darshans/data/"
state_dir = "../../libre/darshan/states/"
params_path(batch, seed, eta, beta, burn_in_time, T, dt, sampling_interval, runs) = string("_batch", 
    batch, "_seed", seed, "_eta", eta, "_beta", beta, "_burn", burn_in_time, "_T", T, "_dt", dt, 
    "_interval", sampling_interval, "_numruns", runs, ".jld")

grad_U(x) = x
F(x) = [x[2], 0]

function drift(x, eta)
    return -grad_U(x) + eta*F(x)
end

function R(x,y)
    return [x[1]*x[2] - y[1]*y[2], norm(x-y)]
end

R_output = 2

function main(coupling, eta, beta, burn_in_time, T, dt, sampling_interval, num_runs, batchs, seed)
    #println(saves)
    X_0 = [1., 0]

    Random.seed!(seed)
    RNG = copy(Random.default_rng())

    N = floor(Int64, T/dt)

    runs_per_batch = num_runs ÷ batchs

    R_traj = zeros(runs_per_batch, N ÷ sampling_interval, 2)
    R_avg = zeros(runs_per_batch, N ÷ sampling_interval, 2)


    if coupling == "indep"
        for batch in 1:batchs
            states = ()
            for j in 1:runs_per_batch
                R_traj[j, :, :], R_avg[j, :, :], state = compute_response(X_0, drift, indep_noise, 
                    R, R_output, eta, beta, burn_in_time, T, dt, sampling_interval, RNG)
                states = (states..., state)
            end
            save(data_dir*"indep_convex2D_resp"*params_path(batch, seed, eta, beta, burn_in_time, 
                T, dt, sampling_interval, runs_per_batch), "R_traj", R_traj, "R_avg", R_avg)
            save(state_dir*"indep_convex2D_resp_state"*params_path(batch, seed, eta, beta, 
                burn_in_time, T, dt, sampling_interval, runs_per_batch), "states", states)
        end
    elseif coupling == "sync"
        for batch in 1:batchs
            states = ()
            for j in 1:runs_per_batch
                R_traj[j, :, :], R_avg[j, :, :], state = compute_response(X_0, drift, sync_noise, 
                    R, R_output, eta, beta, burn_in_time, T, dt, sampling_interval, RNG)
                states = (states..., state)
            end
            save(data_dir*"sync_convex2D_resp"*params_path(batch, seed, eta, beta, burn_in_time, T, 
                dt, sampling_interval, runs_per_batch), "R_traj", R_traj, "R_avg", R_avg)
            save(state_dir*"sync_convex2D_resp_state"*params_path(batch, seed, eta, beta, 
                burn_in_time, T, dt, sampling_interval, runs_per_batch), "states", states)
        end
    elseif coupling == "reflect"
        for batch in 1:batchs
            states = ()
            for j in 1:runs_per_batch
                R_traj[j, :, :], R_avg[j, :, :], state = compute_response(X_0, drift, reflect_noise, 
                    R, R_output, eta, beta, burn_in_time, T, dt, sampling_interval, RNG)
                states = (states..., state)
            end
            save(data_dir*"reflect_convex2D_resp"*params_path(batch, seed, eta, beta, burn_in_time, 
                T, dt, sampling_interval, runs_per_batch), "R_traj", R_traj, "R_avg", R_avg)
            save(state_dir*"reflect_convex2D_resp_state"*params_path(batch, seed, eta, beta, 
                burn_in_time, T, dt, sampling_interval, runs_per_batch), "states", states)
        end
    elseif coupling == "sticky"
        for batch in 1:batchs
            states = ()
            for j in 1:runs_per_batch
                R_traj[j, :, :], R_avg[j, :, :], state = compute_response(X_0, drift, sticky_noise, 
                    R, R_output, eta, beta, burn_in_time, T, dt, sampling_interval, RNG)
                states = (states..., state)
            end
            save(data_dir*"sticky_convex2D_resp"*params_path(batch, seed, eta, beta, burn_in_time, 
                T, dt, sampling_interval, runs_per_batch), "R_traj", R_traj, "R_avg", R_avg)
            save(state_dir*"sticky_convex2D_resp_state"*params_path(batch, seed, eta, beta, 
                burn_in_time, T, dt, sampling_interval, runs_per_batch), "states", states)
        end
    else
        throw(ArgumentError(string(coupling, "not a valid coupling type argument", 
            "\nValid coupling type arguments: \"indep\", \"sync\", \"reflect\", \"sticky\"")))
    end

end


if abspath(PROGRAM_FILE) == @__FILE__
    coupling = ARGS[1]
    eta = parse(Float64, ARGS[2])
    beta = parse(Float64, ARGS[3])
    burn = parse(Float64, ARGS[4])
    T = parse(Float64, ARGS[5])
    dt = parse(Float64, ARGS[6])
    interval = parse(Int64, ARGS[7])
    num_runs = parse(Int64, ARGS[8])
    batchs = parse(Int64, ARGS[9])
    seed = parse(Int64, ARGS[10])
    

    print(string("Started: Two dimensional Convex Response with ", coupling, " coupling\n",
        "eta = ", eta, ", beta = ", beta, ", burn = ", burn, ", T = ", T, ", dt = ", dt, ",\n",  
        "sampling interval = ", interval, ", batchs = ", batchs, ", total runs = ", num_runs, 
        ", seed = ", seed, "\n\n"))

    main(coupling, eta, beta, burn, T, dt, interval, num_runs, batchs, seed)

    print(string("Finished: Two dimensional Convex Response with ", coupling, " coupling\n",
        "eta = ", eta, ", beta = ", beta, ", burn = ", burn, ", T = ", T, ", dt = ", dt, ",\n",  
        "sampling interval = ", interval, ", batchs = ", batchs, ", total runs = ", num_runs, 
        ", seed = ", seed, "\n\n"))
end