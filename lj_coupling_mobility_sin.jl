#2-D cluster of Lennard-Jones particles undergoing sine shear
#with a particle fixed in the origin
#
#Computing mobility, tilt, pressure, and kinetic energy
#
#
#


using Random
using LinearAlgebra 
using Statistics
using JLD

include("lennard_jones.jl")
include("integrators.jl")

const tilt_sharpness::Float64 = 5.


function burnin_step!(Y_n::Array{Float64, 1}, gradient::Array{Float64, 1},
	beta::Float64, dt::Float64, d::Int64, RNG::AbstractRNG)
	"""
	One step of Euler Maruyama for marginal process at equilibrium
	"""
    Y_n .+= -dt .* grad_U!(gradient, Y_n) .+ sqrt(2dt/beta) .* randn(RNG, d)

	return nothing
end

function compute_resp!(R_avg::Array{Float64, 1}, R_traj::Array{Float64, 2}, R_mov_avg::Array{Float64, 2}, 
    X_n::Array{Float64, 1}, gradient_X::Array{Float64, 1}, Y_n::Array{Float64, 1}, 
    gradient_Y::Array{Float64, 1}, M::Float64, tilt_sharpness::Float64, d::Int64, sampling_interval::Int64,
    i::Int64)
    """

    """
    mobility_x = 0.
    mobility_y = 0.
    tilt_x = 0.
    tilt_y = 0.
    pressure_x = -dot(X_n, gradient_X)/2
    pressure_y = -dot(Y_n, gradient_Y)/2
    k_energy_x = dot(gradient_X, gradient_X)/2
    k_energy_y = dot(gradient_Y, gradient_Y)/2
    for j = 1:2:d
        #computing mobility for i-1 step
        mobility_x += sin(pi/M *X_n[j+1]) * gradient_X[j]
        mobility_y += sin(pi/M *Y_n[j+1]) * gradient_Y[j]

        #computing tilt for i-1 step
        tilt_x += tanh(tilt_sharpness * X_n[j]) * tanh(tilt_sharpness * X_n[j+1])
        tilt_y += tanh(tilt_sharpness * Y_n[j]) * tanh(tilt_sharpness * Y_n[j+1])
    end


    R_avg .+= [norm(X_n - Y_n), norm(X_n - Y_n)^2, mobility_x - mobility_y, mobility_x, mobility_y, 
    tilt_x - tilt_y, tilt_x, tilt_y, pressure_x - pressure_y, pressure_x, pressure_y, k_energy_x - k_energy_y, k_energy_x, k_energy_y]

    if (i-1)%sampling_interval == 0
        R_traj[(i-1) ÷ sampling_interval, :] .= [norm(X_n - Y_n), norm(X_n - Y_n)^2, mobility_x - mobility_y, mobility_x, mobility_y, tilt_x - tilt_y, tilt_x, tilt_y, pressure_x - pressure_y, pressure_x, pressure_y, k_energy_x - k_energy_y, k_energy_x, k_energy_y]
        R_mov_avg[(i-1) ÷ sampling_interval, :] .= R_avg ./ (i-1)
    end

    return nothing
end

function compute_mobdist(X_0::Array{Float64, 1}, coupled_noise::Function, eta::Float64, beta::Float64, 
    burn_in_time::Float64, T::Float64, dt::Float64, sampling_interval::Int64, RNG::AbstractRNG = copy(Random.default_rng()))
	"""
	Computing mobility in a way to avoid too many evaluations of grad_U 
	"""
	N = floor(Int, T/dt)
    dt = T/N

    Y_n = copy(X_0)
    d = length(X_0)
    
    burn_k = floor(Int, burn_in_time/dt)
	gradient_X = zeros(d)
    gradient_Y = zeros(d)
    G = zeros(d)

    for i = 1:burn_k
        burnin_step!(Y_n, gradient_Y, beta, dt, d, RNG)
    end

    R_traj = zeros(N ÷ sampling_interval, 14)
    R_mov_avg = zeros(N ÷ sampling_interval, 14)
    R_avg = zeros(14)
   

    X_n = copy(Y_n)

    #first step
    grad_U!(gradient_X, X_n)
    grad_U!(gradient_Y, Y_n)
    for j = 1:2:d
        #linear shear forcing
        X_n[j] += eta*dt*X_n[j+1]
    end
    X_n .+= -dt .* gradient_X
    Y_n .+= -dt .* gradient_Y
    stochastic_step!(X_n, Y_n, G, coupled_noise, d, dt, beta, RNG)

    for i = 2:(N+1)
        #compute once the gradient 
        grad_U!(gradient_X, X_n)
        grad_U!(gradient_Y, Y_n)

        #in effect we compute the response for the i-1 step
        compute_resp!(R_avg, R_traj, R_mov_avg, X_n, gradient_X, Y_n, gradient_Y, M, tilt_sharpness, d, sampling_interval, i)

    
        #evolve drift part of dynamics
        for j = 1:2:d
            #sin shear forcing
            X_n[j] += eta*dt*sin(pi/M * X_n[j+1])
        end
        X_n .+= -dt .* gradient_X
        Y_n .+= -dt .* gradient_Y

        #evolve stochastic part of dynamics
        stochastic_step!(X_n, Y_n, G, coupled_noise, d, dt, beta, RNG)
    end

    RNG_state = ntuple(fieldcount(typeof(RNG))) do i
		getfield(RNG, i)
	end

    return R_traj, R_mov_avg, ((X_n, Y_n), RNG_state)
end

coupled_noises = Dict([("indep", indep_noise), ("sync", sync_noise), ("reflect", reflect_noise), ("sticky", sticky_noise)])


function main(coupling, L, eta, beta, burn_in_time, T, dt, sampling_interval, seed; save_data = true)
    Random.seed!(seed)
    RNG = copy(Random.default_rng())

    #println(saves)
    X_0 = getX_0(L)
    distresp_traj, distresp_avg, state = compute_mobdist(X_0, coupled_noises[coupling], eta, beta, burn_in_time, T, 
    dt, sampling_interval, RNG)

    if save_data
        save(string("../../libre/darshans/data/lj_", coupling, "_coupling_distmobilitypressuretilt_sinshear_seed", seed, "_L", L, "_eta", 
            eta, "_beta", beta, "_T", T, "_dt", dt, "_burn", burn_in_time, "_interval", sampling_interval, 
            ".jld"), "distresp_traj", distresp_traj, "distresp_avg", distresp_avg)
        save(string("../../libre/darshans/states/lj_", coupling, "_coupling_distmobilitypressuretilt_sinshear_state_seed", seed, "_L", L, "_eta", 
            eta, "_beta", beta, "_T", T, "_dt", dt, "_burn", burn_in_time, "_interval", sampling_interval, 
            ".jld"), "state", state)
    else
        return distresp_traj, distresp_avg, state
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
    seed = parse(Int64, ARGS[9])

    print(string("Started: Lennard-Jones mobility, pressure, and tilt with sine shear\n, coupling = ", 
        coupling, ", L = ", L, ",\n",
        "eta = ", eta, ", beta = ", beta, ",\n", 
        ", dt = ", dt, ", T = ", T, ", burn-in = ", burn,", interval = ", interval, ",\n", 
        ", seed = ", seed, "\n\n"))
    flush(stdout)
    main(coupling, L, eta, beta, burn, T, dt, interval, seed)
    print(string("Finished: Lennard-Jones mobility, pressure, and tilt with sine shear\n, coupling = ", 
        coupling, ", L = ", L, ",\n",
        "eta = ", eta, ", beta = ", beta, ",\n", 
        ", dt = ", dt, ", T = ", T, ", burn-in = ", burn,", interval = ", interval, ",\n", 
        ", seed = ", seed, "\n\n"))
    flush(stdout)
end