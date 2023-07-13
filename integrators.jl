using Random
using LinearAlgebra



function integration_step!(Y_n::Array{Float64, 1}, drift::Function, 
	beta::Float64, dt::Float64, d::Int64, RNG::AbstractRNG)
	"""
	One step of Euler Maruyama for marginal process at equilibrium
	"""
    Y_n .+= dt .* drift(Y_n, 0.) .+ sqrt(2*dt/beta) .* randn(RNG, d)
	
	return nothing
end

#For the coupled processes we break up the Euler-Maruyama integration into two function
#one for the deterministic part and one for the stochastic part

function deterministic_step!(X_n::Array{Float64, 1}, Y_n::Array{Float64, 1},
	drift::Function, eta::Float64, dt::Float64)
	"""
	Deterministic part of one step of Euler-Maruyama for the coupled process
	"""
	X_n .+= dt * drift(X_n, eta) 
    Y_n .+= dt * drift(Y_n, 0.)

	return nothing
end

##WORK IN PROGRESS

function integration_step!(Y_n::Array{Float64, 1}, gradient::Array{Float64, 1}, drift::Function, 
	beta::Float64, dt::Float64, d::Int64, RNG::AbstractRNG)
	"""
	One step of Euler Maruyama for marginal process at equilibrium
	"""
    Y_n .+= dt .* drift(gradient, Y_n, 0.) .+ sqrt(2dt/beta) .* randn(RNG, d)

	return nothing
end
function deterministic_step!(X_n::Array{Float64, 1}, Y_n::Array{Float64, 1},
	gradient::Array{Float64, 1}, drift!::Function, eta::Float64, dt::Float64)
	"""
	Deterministic part of one step of Euler-Maruyama for the coupled process
	"""
	X_n .+= dt * drift!(gradient, X_n, eta) 
    Y_n .+= dt * drift!(gradient, Y_n, 0.)

	return nothing
end

function stochastic_step!(X_n::Array{Float64, 1}, Y_n::Array{Float64, 1}, G::Array, 
	coupled_noise::Function, d::Int64, dt::Float64, beta::Float64, RNG::AbstractRNG, 
	tol::Float64 = 1e-14)
	"""
	Stochastic part of one step of Euler-Maruyama for the coupled process
	"""
	randn!(RNG, G)

	Y_n .+= sqrt(2*dt/beta) .* coupled_noise(G, X_n, Y_n, d, dt, beta, RNG, tol) 
	X_n .+= sqrt(2*dt/beta) .* G

	return nothing
end


# The for types of coupled driving noise for the reference process:
# 1) Independent coupling (Y_n's driving noise is independent of X_n's)
# 2) Synchronous coupling (Y_n and X_n are driven by the same noise)
# 3) Reflection coupling (Y_n is driven by X_n's noise reflected across their seperating plane)
# 4) Sticky coupling

function indep_noise(G::Array{Float64, 1}, X_n::Array{Float64, 1}, Y_n::Array{Float64, 1}, d::Int64, 
	dt::Float64, beta::Float64, RNG::AbstractRNG, tol::Float64 = 1e-14)
	"""
	Independent coupling of driving noise
	"""
	return randn(RNG, d) 
end

function sync_noise(G::Array{Float64, 1}, X_n::Array{Float64, 1}, Y_n::Array{Float64, 1}, d::Int64, 
	dt::Float64, beta::Float64, RNG::AbstractRNG, tol::Float64 = 1e-14)
	"""
	Synchronous coupling of driving noise
	"""
	return G 
end

function reflect_noise(G::Array{Float64, 1}, X_n::Array{Float64, 1}, Y_n::Array{Float64, 1}, d::Int64, 
	dt::Float64, beta::Float64, RNG::AbstractRNG, tol::Float64 = 1e-14)
	"""
	Reflection coupling of driving noise
	"""
	u = norm(X_n - Y_n) < tol ? 0 : (X_n - Y_n)/norm(X_n - Y_n) #to avoid a Zero divison error when X_n == Y_n 
	P = I - 2 * (u * u')
	return P * G
end

normpdf(x, d) = exp(-dot(x,x)/2)/(2*pi)^(d/2) #standard d-dimensional Gaussian density
function sticky_noise(G::Array{Float64, 1}, X_n::Array{Float64, 1}, Y_n::Array{Float64, 1}, d::Int64,
	dt::Float64, beta::Float64, RNG::AbstractRNG, tol::Float64 = 1e-14)
	"""
	Maximal-reflection coupling of driving noise following algorithm 3 from the chapter 2 of Pierre 
	Jacob's course notes
	"""
    z = sqrt(beta/(2*dt)) * (X_n - Y_n)
    U = rand(RNG)

    if U < normpdf(G + z, d)/normpdf(G, d)
    	return G + z
    else
        u = z/norm(z)
        P = I - 2 * (u * u')
        return P * G
    end
end


function compute_traj(X_0::Array{Float64, 1}, drift::Function, coupled_noise::Function, eta::Float64,
	beta::Float64, burn_in_time::Float64, T::Float64, dt::Float64, sampling_interval::Int64, 
	RNG::AbstractRNG = copy(Random.default_rng()), tol::Float64 = 1e-14)
	"""
	Record trajectory of coupled process at given sampling_interval
	"""
	N = floor(Int, T/dt)
    dt = T/N

    Y_n = copy(X_0)
    d = length(X_0)

    burn_k = floor(Int, burn_in_time/dt)
    G = zeros(d)

    for i = 1:burn_k
    	integration_step!(Y_n, drift, beta, dt, d, RNG)
    end

    X_traj = zeros(N÷sampling_interval, d)
    Y_traj = zeros(N÷sampling_interval, d)

    X_n = copy(Y_n)

    for i = 1:N
        deterministic_step!(X_n, Y_n, drift, eta, dt)
        stochastic_step!(X_n, Y_n, G, coupled_noise, d, dt, beta, RNG)

        if i%sampling_interval == 0
	        X_traj[i ÷ sampling_interval, :] .= X_n
	        Y_traj[i ÷ sampling_interval, :] .= Y_n
	    end
    end

    RNG_state = ntuple(fieldcount(typeof(RNG))) do i
		getfield(RNG, i)
	end

    return X_traj, Y_traj, ((X_n, Y_n), RNG_state)
end

function compute_response(X_0::Array{Float64, 1}, drift::Function, coupled_noise::Function, R::Function, 
	output_dim::Int64, eta::Float64, beta::Float64, burn_in_time::Float64, T::Float64, dt::Float64, 
	sampling_interval::Int64, RNG::AbstractRNG = copy(Random.default_rng()), tol::Float64 = 1e-14)
	"""
	Record trajectory and cumulative average of specified observable R at specified sampling_interval 
	"""
	N = floor(Int, T/dt)
    dt = T/N

    Y_n = copy(X_0)
    d = length(X_0)
    
    burn_k = floor(Int, burn_in_time/dt)
	gradient = zeros(d)
    G = zeros(d)

    for i = 1:burn_k
        integration_step!(Y_n, drift, beta, dt, d, RNG)
    end

    R_traj = zeros(N ÷ sampling_interval, output_dim)
    R_mov_avg = zeros(N ÷ sampling_interval, output_dim)
    R_avg = zeros(output_dim)

    X_n = copy(Y_n)

    for i = 1:N
        deterministic_step!(X_n, Y_n, drift, eta, dt)
        stochastic_step!(X_n, Y_n, G, coupled_noise, d, dt, beta, RNG)

        R_avg .+= R(X_n, Y_n)


        if i%sampling_interval == 0
	        R_traj[i ÷ sampling_interval, :] .= R(X_n, Y_n)
	        R_mov_avg[i ÷ sampling_interval, :] .= R_avg ./ i
	    end
    end

    RNG_state = ntuple(fieldcount(typeof(RNG))) do i
		getfield(RNG, i)
	end

    return R_traj, R_mov_avg, ((X_n, Y_n), RNG_state)
end

function compute_response2(X_0::Array{Float64, 1}, drift!::Function, coupled_noise::Function, R::Function, 
	output_dim::Int64, eta::Float64, beta::Float64, burn_in_time::Float64, T::Float64, dt::Float64, 
	sampling_interval::Int64, RNG::AbstractRNG = copy(Random.default_rng()), tol::Float64 = 1e-14)
	"""
	Record trajectory and cumulative average of specified observable R at specified sampling_interval 
	"""
	N = floor(Int, T/dt)
    dt = T/N

    Y_n = copy(X_0)
    d = length(X_0)
    
    burn_k = floor(Int, burn_in_time/dt)
	gradient = zeros(d)
    G = zeros(d)

    for i = 1:burn_k
        integration_step!(Y_n, gradient, drift!, beta, dt, d, RNG)
    end

    R_traj = zeros(N ÷ sampling_interval, output_dim)
    R_mov_avg = zeros(N ÷ sampling_interval, output_dim)
    R_avg = zeros(output_dim)

    X_n = copy(Y_n)

    for i = 1:N
        deterministic_step!(X_n, Y_n, gradient, drift!, eta, dt)
        stochastic_step!(X_n, Y_n, G, coupled_noise, d, dt, beta, RNG)

        R_avg .+= R(X_n, Y_n)


        if i%sampling_interval == 0
	        R_traj[i ÷ sampling_interval, :] .= R(X_n, Y_n)
	        R_mov_avg[i ÷ sampling_interval, :] .= R_avg ./ i
	    end
    end

    RNG_state = ntuple(fieldcount(typeof(RNG))) do i
		getfield(RNG, i)
	end

    return R_traj, R_mov_avg, ((X_n, Y_n), RNG_state)
end


function compute_response_IC_diff(X_0::Array{Float64, 1}, drift::Function, coupled_noise::Function, R::Function, 
	output_dim::Int64, eta::Float64, beta::Float64, burn_in_time::Float64, T::Float64, dt::Float64, 
	sampling_interval::Int64, RNG::AbstractRNG = copy(Random.default_rng()), tol::Float64 = 1e-14)
	"""
	Record trajectory and cumulative average of specified observable R at specified sampling_interval 
	"""
	N = floor(Int, T/dt)
    dt = T/N

    Y_n = copy(X_0)
    d = length(X_0)
    
    burn_k = floor(Int, burn_in_time/dt)
	gradient = zeros(d)
    G = zeros(d)

    for i = 1:burn_k
        integration_step!(Y_n, drift, beta, dt, d, RNG)
    end

    R_traj = zeros(N ÷ sampling_interval, output_dim)
    R_mov_avg = zeros(N ÷ sampling_interval, output_dim)
    R_avg = zeros(output_dim)

    X_n = copy(Y_n)

    for i = 1:burn_k
    	integration_step!(X_n, drift, beta, dt, d, RNG)
    end

    X_0 = copy(X_n)
    Y_0 = copy(Y_n)

    for i = 1:N
        deterministic_step!(X_n, Y_n, drift, eta, dt)
        stochastic_step!(X_n, Y_n, G, coupled_noise, d, dt, beta, RNG)

        R_avg .+= R(X_n, Y_n)


        if i%sampling_interval == 0
	        R_traj[i ÷ sampling_interval, :] .= R(X_n, Y_n)
	        R_mov_avg[i ÷ sampling_interval, :] .= R_avg ./ i
	    end
    end

    RNG_state = ntuple(fieldcount(typeof(RNG))) do i
		getfield(RNG, i)
	end

    return R_traj, R_mov_avg, ((X_n, Y_n), RNG_state, (X_0, Y_0))
end

