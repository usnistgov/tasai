# sunny_square_lattice.jl
#
# Square lattice ferromagnet with nearest-neighbor (J1) and 
# next-nearest-neighbor (J2) interactions for TAS-AI integration.
#
# Hamiltonian:
#   H = -J1 Σ_<i,j> Si·Sj - J2 Σ_<<i,j>> Si·Sj - D Σ_i (Siz)²
#
# - J1 > 0: FM nearest neighbor (in-plane)
# - J2: next-nearest neighbor (diagonal, can be FM or AFM)
# - D > 0: easy-axis anisotropy along c
# - No c-axis coupling (2D magnetic layers)
#
# Usage from Julia:
#   include("sunny_square_lattice.jl")
#   model = create_model(J1=1.0, J2=0.1, D=0.05, S=1.0)
#   energies, intensities = calc_sqw(model, Qpoints, ω_range)
#
# Usage from Python (via JSON interface):
#   julia sunny_square_lattice.jl --calc-sqw --params params.json --output result.json

using Sunny
using LinearAlgebra
using JSON
using ArgParse

"""
Create the square lattice crystal structure.
Tetragonal unit cell with lattice parameters a, b, c.
"""
function create_crystal(; a=4.0, c=6.0)
    # Tetragonal P4/mmm (space group 123)
    # Single magnetic site at origin
    latvecs = lattice_vectors(a, a, c, 90, 90, 90)
    positions = [[0, 0, 0]]
    types = ["Fe"]
    
    Crystal(latvecs, positions; types=types)
end

"""
Create the spin system with exchange interactions and anisotropy.

Parameters:
- J1: Nearest-neighbor exchange (meV), positive = FM
- J2: Next-nearest-neighbor exchange (meV)  
- D: Single-ion anisotropy (meV), positive = easy c-axis
- S: Spin quantum number (e.g., 1.0, 1.5, 2.0)
- dims: Supercell dimensions for spin wave calculation
"""
function create_model(; J1=1.0, J2=0.0, D=0.05, S=1.0, a=4.0, c=6.0, dims=(1,1,1))
    cryst = create_crystal(a=a, c=c)
    
    # Create spin system
    # Using Sr=S for spin quantum number
    sys = System(cryst, dims, [SpinInfo(1, S=S, g=2.0)], :dipole)
    
    # Nearest-neighbor bonds (along a and b axes)
    # Bond from (0,0,0) to (1,0,0) and (0,1,0)
    bond_nn_a = Bond(1, 1, [1, 0, 0])  # Along a
    bond_nn_b = Bond(1, 1, [0, 1, 0])  # Along b
    
    # Next-nearest-neighbor bonds (diagonals in ab-plane)
    # Bond from (0,0,0) to (1,1,0) and (1,-1,0)
    bond_nnn_pp = Bond(1, 1, [1, 1, 0])   # Diagonal ++
    bond_nnn_pm = Bond(1, 1, [1, -1, 0])  # Diagonal +-
    
    # Set exchange interactions
    # Heisenberg exchange: J * Si·Sj = J * (Sx*Sx + Sy*Sy + Sz*Sz)
    # In Sunny, set_exchange! uses the convention H = Σ Si·J·Sj
    # For isotropic Heisenberg: J_matrix = J * I (identity)
    # Note: Sunny uses H = +J*Si·Sj, so FM needs J < 0
    
    J1_matrix = -J1 * I(3)  # Negative for FM convention
    J2_matrix = -J2 * I(3)
    
    set_exchange!(sys, J1_matrix, bond_nn_a)
    set_exchange!(sys, J1_matrix, bond_nn_b)
    
    if abs(J2) > 1e-10
        set_exchange!(sys, J2_matrix, bond_nnn_pp)
        set_exchange!(sys, J2_matrix, bond_nnn_pm)
    end
    
    # Single-ion anisotropy: -D * Sz^2
    # Easy axis along c for D > 0
    # In Sunny: set_onsite_coupling!(sys, S -> -D*S[3]^2, 1)
    if abs(D) > 1e-10
        set_onsite_coupling!(sys, S -> -D*S[3]^2, 1)
    end
    
    # Initialize to FM ground state (all spins along +c)
    for site in eachsite(sys)
        set_dipole!(sys, [0, 0, 1], site)
    end
    
    return sys
end

"""
Calculate spin wave dispersion using linear spin wave theory.

Returns SpinWaveTheory object for subsequent S(Q,ω) calculations.
"""
function create_swt(sys)
    # Create SpinWaveTheory object
    swt = SpinWaveTheory(sys)
    return swt
end

"""
Calculate S(Q,ω) at specified Q points and energy range.

Parameters:
- swt: SpinWaveTheory object
- Qpoints: Array of Q vectors in r.l.u. [[H1,K1,L1], [H2,K2,L2], ...]
- ω_range: Tuple (ω_min, ω_max, n_ω) for energy binning
- η: Energy broadening (half-width, meV)

Returns:
- energies: Array of energy values
- intensities: 2D array [n_Q, n_ω] of S(Q,ω)
"""
function calc_sqw(swt, Qpoints; ω_min=0.0, ω_max=10.0, n_ω=100, η=0.1)
    energies = range(ω_min, ω_max, length=n_ω)
    
    # Calculate intensities for each Q point
    n_Q = size(Qpoints, 1)
    intensities = zeros(n_Q, n_ω)
    
    # Energy broadening kernel
    kernel = lorentzian(fwhm=2η)
    
    for (i, Q) in enumerate(eachrow(Qpoints))
        q_vec = Vec3(Q...)
        
        # Get spin wave energies and intensities at this Q
        disp = dispersion(swt, [q_vec])
        Sqw = intensities_broadened(swt, [q_vec], energies, kernel)
        
        # Sum over all spin wave modes
        intensities[i, :] = dropdims(sum(Sqw, dims=1), dims=1)
    end
    
    return collect(energies), intensities
end

"""
Calculate dispersion along a Q path.

Parameters:
- swt: SpinWaveTheory object  
- q_path: Array of Q points defining the path
- n_points: Number of points along path

Returns:
- q_distances: Cumulative distance along path
- energies: Spin wave energies at each point [n_points, n_modes]
"""
function calc_dispersion(swt, q_path; n_points=100)
    # Interpolate path
    path_points = []
    for i in 1:(length(q_path)-1)
        q1, q2 = q_path[i], q_path[i+1]
        for t in range(0, 1, length=n_points÷(length(q_path)-1))
            push!(path_points, (1-t)*q1 + t*q2)
        end
    end
    
    # Calculate dispersion
    q_vecs = [Vec3(q...) for q in path_points]
    disp = dispersion(swt, q_vecs)
    
    # Calculate path distances
    distances = [0.0]
    for i in 2:length(path_points)
        d = norm(path_points[i] - path_points[i-1])
        push!(distances, distances[end] + d)
    end
    
    return distances, disp
end

"""
Calculate intensity at a single (Q, E) point with resolution convolution.

This is the main function called by TAS-AI for simulated measurements.

Parameters:
- sys: Spin system
- H, K, L: Q-point in r.l.u.
- E: Energy transfer (meV)
- ΔE: Energy resolution FWHM (meV)
- ΔQ: Q resolution FWHM (r.l.u.)

Returns:
- intensity: S(Q,E) convolved with resolution
"""
function calc_intensity(sys, H, K, L, E; ΔE=0.5, ΔQ=0.02)
    swt = SpinWaveTheory(sys)
    
    # Simple Gaussian convolution over Q and E
    # Sample nearby Q points
    n_Q_samples = 5
    n_E_samples = 11
    
    intensity = 0.0
    weight_sum = 0.0
    
    E_samples = range(E - 2ΔE, E + 2ΔE, length=n_E_samples)
    
    for dH in range(-2ΔQ, 2ΔQ, length=n_Q_samples)
        for dK in range(-2ΔQ, 2ΔQ, length=n_Q_samples)
            Q = Vec3(H + dH, K + dK, L)
            Q_weight = exp(-(dH^2 + dK^2)/(2*ΔQ^2))
            
            # Get spin wave response at this Q
            kernel = lorentzian(fwhm=ΔE)
            Sqw = intensities_broadened(swt, [Q], E_samples, kernel)
            
            # Find intensity at target energy
            E_idx = argmin(abs.(collect(E_samples) .- E))
            I_Q = sum(Sqw[:, E_idx])
            
            intensity += I_Q * Q_weight
            weight_sum += Q_weight
        end
    end
    
    return intensity / weight_sum
end

"""
JSON interface for Python integration.

Reads parameters from JSON, runs calculation, writes results to JSON.
"""
function run_from_json(params_file::String, output_file::String)
    # Read parameters
    params = JSON.parsefile(params_file)
    
    # Create model
    J1 = get(params, "J1", 1.0)
    J2 = get(params, "J2", 0.0)
    D = get(params, "D", 0.05)
    S = get(params, "S", 1.0)
    a = get(params, "a", 4.0)
    c = get(params, "c", 6.0)
    
    sys = create_model(J1=J1, J2=J2, D=D, S=S, a=a, c=c)
    
    # Determine calculation type
    calc_type = get(params, "calc_type", "intensity")
    
    result = Dict()
    
    if calc_type == "intensity"
        # Single point intensity
        H = get(params, "H", 0.0)
        K = get(params, "K", 0.0)
        L = get(params, "L", 0.0)
        E = get(params, "E", 1.0)
        ΔE = get(params, "dE", 0.5)
        ΔQ = get(params, "dQ", 0.02)
        
        I = calc_intensity(sys, H, K, L, E, ΔE=ΔE, ΔQ=ΔQ)
        result["intensity"] = I
        result["H"] = H
        result["K"] = K
        result["L"] = L
        result["E"] = E
        
    elseif calc_type == "sqw"
        # S(Q,ω) map
        Qpoints = get(params, "Qpoints", [[0.0, 0.0, 0.0]])
        ω_min = get(params, "w_min", 0.0)
        ω_max = get(params, "w_max", 10.0)
        n_ω = get(params, "n_w", 100)
        η = get(params, "eta", 0.1)
        
        swt = SpinWaveTheory(sys)
        Q_array = reduce(vcat, transpose.(Qpoints))
        energies, intensities = calc_sqw(swt, Q_array, 
                                         ω_min=ω_min, ω_max=ω_max, 
                                         n_ω=n_ω, η=η)
        
        result["energies"] = energies
        result["intensities"] = intensities
        result["Qpoints"] = Qpoints
        
    elseif calc_type == "dispersion"
        # Dispersion along path
        q_path = get(params, "q_path", [[0,0,0], [0.5,0.5,0]])
        n_points = get(params, "n_points", 100)
        
        swt = SpinWaveTheory(sys)
        path_array = [Float64.(q) for q in q_path]
        distances, disp = calc_dispersion(swt, path_array, n_points=n_points)
        
        result["distances"] = distances
        result["dispersion"] = disp
        result["q_path"] = q_path
    end
    
    # Add model parameters to result
    result["params"] = Dict("J1"=>J1, "J2"=>J2, "D"=>D, "S"=>S, "a"=>a, "c"=>c)
    
    # Write output
    open(output_file, "w") do f
        JSON.print(f, result, 2)
    end
    
    return result
end

# Command-line interface
function main()
    s = ArgParseSettings(description="Sunny.jl square lattice FM calculations")
    
    @add_arg_table! s begin
        "--params"
            help = "Input parameters JSON file"
            arg_type = String
            required = true
        "--output"
            help = "Output results JSON file"
            arg_type = String
            required = true
    end
    
    args = parse_args(s)
    run_from_json(args["params"], args["output"])
end

# Run if called as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
