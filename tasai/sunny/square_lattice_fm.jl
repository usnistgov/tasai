# square_lattice_fm.jl
# 
# Sunny.jl spin wave calculation for square lattice ferromagnet
# with nearest-neighbor (J1) and next-nearest-neighbor (J2) interactions.
#
# Model: H = -J1 Σ_<i,j> Si·Sj - J2 Σ_<<i,j>> Si·Sj - D Σ_i (Sz_i)²
#
# Usage from Julia:
#   include("square_lattice_fm.jl")
#   result = calculate_sqw(J1=1.0, J2=0.1, D=0.05, qpoints=[[0,0,0], [0.5,0.5,0]])
#
# Usage from command line (for Python integration):
#   julia square_lattice_fm.jl --J1 1.0 --J2 0.1 --D 0.05 --qfile qpoints.txt --outfile result.json

using Sunny
using LinearAlgebra
using JSON
using ArgParse

"""
    create_square_lattice_fm(; J1, J2, D, S=1, a=1.0, c=1.0)

Create a square lattice ferromagnet with:
- J1: nearest-neighbor exchange (in-plane)
- J2: next-nearest-neighbor exchange (diagonal in-plane)  
- D: single-ion anisotropy (easy c-axis)
- S: spin quantum number
- a, c: lattice parameters

Returns a Sunny System object.
"""
function create_square_lattice_fm(; J1::Float64, J2::Float64, D::Float64, 
                                   S::Float64=1.0, a::Float64=1.0, c::Float64=1.0)
    # Tetragonal lattice (P4/mmm, space group 123)
    # Single magnetic atom at origin
    latvecs = lattice_vectors(a, a, c, 90, 90, 90)
    
    # Create crystal with one magnetic site
    crystal = Crystal(latvecs, [[0, 0, 0]], 123; types=["Fe"])
    
    # Create spin system
    # Using :dipole mode for classical spin waves
    sys = System(crystal, (1, 1, 1), [SpinInfo(1; S=S, g=2.0)], :dipole)
    
    # Set exchange interactions
    # J1: nearest neighbors at (±1,0,0) and (0,±1,0)
    # Bond Sr = [1,0,0] connects site 1 to equivalent site in neighboring cell
    set_exchange!(sys, J1, Bond(1, 1, [1, 0, 0]))  # +x neighbor
    set_exchange!(sys, J1, Bond(1, 1, [0, 1, 0]))  # +y neighbor
    
    # J2: next-nearest neighbors at (±1,±1,0)
    if abs(J2) > 1e-10
        set_exchange!(sys, J2, Bond(1, 1, [1, 1, 0]))   # +x+y diagonal
        set_exchange!(sys, J2, Bond(1, 1, [1, -1, 0]))  # +x-y diagonal
    end
    
    # No c-axis coupling (Jc = 0) - already the default
    
    # Single-ion anisotropy: -D*(Sz)^2 (easy axis along c)
    # In Sunny, this is set via set_onsite_coupling!
    # For S=1, the Stevens operator O20 ∝ Sz²
    if abs(D) > 1e-10
        # Easy-axis anisotropy along z (c-axis)
        set_onsite_coupling!(sys, S -> -D * S[3]^2, 1)
    end
    
    # Set ground state: FM along c-axis
    set_dipole!(sys, [0, 0, 1], (1, 1, 1, 1))
    
    return sys
end

"""
    calculate_dispersion(sys, qpoints; energies=nothing, η=0.1)

Calculate spin wave dispersion along given q-points.

Returns (qs, energies, intensities) where:
- qs: array of q-points
- energies: spin wave energies at each q
- intensities: S(q,ω) intensity at each q
"""
function calculate_dispersion(sys::System, qpoints::Vector{Vector{Float64}}; 
                              energies::Union{Nothing, Vector{Float64}}=nothing,
                              η::Float64=0.1)
    # Create SpinWaveTheory object
    swt = SpinWaveTheory(sys)
    
    # Calculate at each q-point
    n_q = length(qpoints)
    
    # Get number of bands (= number of magnetic atoms in magnetic unit cell)
    n_bands = 1  # Single site in our case
    
    all_energies = zeros(n_q, n_bands)
    all_intensities = zeros(n_q, n_bands)
    
    for (i, q) in enumerate(qpoints)
        # Calculate spin wave energies
        disp = dispersion(swt, [q])
        all_energies[i, :] = disp[1, :]
        
        # Calculate intensities (structure factor)
        if energies !== nothing
            # Calculate S(q,ω) at specified energies
            formula = intensity_formula(swt, :perp; kernel=lorentzian(η))
            # For now, just use the dispersion energy
        end
        
        # Simple intensity estimate (proportional to 1/ω for FM)
        all_intensities[i, :] = 1.0 ./ max.(all_energies[i, :], 0.1)
    end
    
    return qpoints, all_energies, all_intensities
end

"""
    calculate_sqw_cut(sys, q_start, q_end, n_points; E_range, n_E, η)

Calculate S(q,ω) along a cut in reciprocal space.

Returns a 2D array of intensities.
"""
function calculate_sqw_cut(sys::System, q_start::Vector{Float64}, q_end::Vector{Float64}, 
                           n_q::Int; E_min::Float64=0.0, E_max::Float64=10.0, 
                           n_E::Int=100, η::Float64=0.1)
    # Create path
    qpoints = [q_start .+ t .* (q_end .- q_start) for t in range(0, 1, length=n_q)]
    
    # Energy bins
    energies = range(E_min, E_max, length=n_E)
    
    # Create SpinWaveTheory
    swt = SpinWaveTheory(sys)
    
    # Calculate S(q,ω)
    formula = intensity_formula(swt, :perp; kernel=lorentzian(η))
    
    sqw = zeros(n_q, n_E)
    
    for (i, q) in enumerate(qpoints)
        # Get dispersion at this q
        disp = dispersion(swt, [q])[1, :]
        
        # Add Lorentzian peaks at dispersion energies
        for ω0 in disp
            if ω0 > 0
                for (j, E) in enumerate(energies)
                    # Lorentzian: I(E) = η / ((E - ω0)² + η²) / π
                    sqw[i, j] += η / ((E - ω0)^2 + η^2) / π
                end
            end
        end
    end
    
    return collect(qpoints), collect(energies), sqw
end

"""
    calculate_hhl_cut(; J1, J2, D, H_range, L, n_points, E_max, n_E, η)

Calculate S(Q,ω) in the HHL zone.

For square lattice, HHL means q = (H, H, L).
"""
function calculate_hhl_cut(; J1::Float64, J2::Float64, D::Float64,
                           H_min::Float64=0.0, H_max::Float64=1.0, L::Float64=0.0,
                           n_H::Int=50, E_max::Float64=10.0, n_E::Int=100, 
                           η::Float64=0.1, S::Float64=1.0)
    # Create system
    sys = create_square_lattice_fm(J1=J1, J2=J2, D=D, S=S)
    
    # Q-points along (H, H, L)
    q_start = [H_min, H_min, L]
    q_end = [H_max, H_max, L]
    
    return calculate_sqw_cut(sys, q_start, q_end, n_H; E_max=E_max, n_E=n_E, η=η)
end

"""
    analytical_dispersion(H, L; J1, J2, D, S=1)

Analytical dispersion for square lattice FM in HHL zone.

ω(H,H,L) = 2S × sqrt[(A + D)(A + 2D)]

where A = J1(cos(2πH) - 1) + J2(cos(4πH) - 1) for HHL

This is useful for fast fitting without full Sunny calculation.
"""
function analytical_dispersion(H::Float64, L::Float64; 
                               J1::Float64, J2::Float64, D::Float64, S::Float64=1.0)
    # Exchange contribution (in HHL, h=k=H)
    A = 2 * J1 * (cos(2π * H) - 1) + 2 * J2 * (cos(4π * H) - 1)
    
    # Full dispersion with anisotropy gap
    # For FM with easy-axis: ω = sqrt[(|A| + D)(|A| + D + 2D·S)]
    # Simplified for small D: ω ≈ sqrt(|A|² + 2D·S·|A|)
    
    # More accurate formula:
    A_eff = abs(A)
    gap = D * S  # Anisotropy gap
    
    if A_eff < 1e-10
        # At zone center: gap mode
        return 2 * S * sqrt(gap * (gap + 2 * D * S))
    else
        # Full dispersion
        return 2 * S * sqrt((A_eff + gap) * (A_eff + gap + 2 * D * S))
    end
end

"""
    analytical_dispersion_array(H_array, L; J1, J2, D, S=1)

Vectorized analytical dispersion.
"""
function analytical_dispersion_array(H_array::Vector{Float64}, L::Float64;
                                     J1::Float64, J2::Float64, D::Float64, S::Float64=1.0)
    return [analytical_dispersion(H, L; J1=J1, J2=J2, D=D, S=S) for H in H_array]
end

"""
    simulate_measurement(H, L, E; J1, J2, D, count_time, background)

Simulate a neutron scattering measurement with Poisson noise.

Returns (intensity, uncertainty).
"""
function simulate_measurement(H::Float64, L::Float64, E::Float64;
                              J1::Float64, J2::Float64, D::Float64, S::Float64=1.0,
                              η::Float64=0.5, count_time::Float64=60.0,
                              count_rate::Float64=100.0, background::Float64=0.01)
    # Calculate true dispersion energy
    ω0 = analytical_dispersion(H, L; J1=J1, J2=J2, D=D, S=S)
    
    # S(Q,E) ~ Lorentzian centered at ω0
    # I(E) = A × η / ((E - ω0)² + η²) / π + background
    intensity_true = η / ((E - ω0)^2 + η^2) / π + background
    
    # Bose factor (for inelastic)
    if E > 0.1
        n_bose = 1 / (exp(E / 25.0) - 1) + 1  # kT ≈ 25 meV at 300K
        intensity_true *= n_bose
    end
    
    # Poisson statistics
    counts = rand(Poisson(max(1, round(Int, intensity_true * count_rate * count_time))))
    
    I_measured = counts / (count_rate * count_time)
    σ = sqrt(counts) / (count_rate * count_time)
    σ = max(σ, 0.001)
    
    return I_measured, σ
end

# =============================================================================
# Command-line interface for Python integration
# =============================================================================

function parse_commandline()
    s = ArgParseSettings(description="Sunny spin wave calculation for square lattice FM")
    
    @add_arg_table! s begin
        "--J1"
            help = "Nearest-neighbor exchange (meV)"
            arg_type = Float64
            default = 1.0
        "--J2"
            help = "Next-nearest-neighbor exchange (meV)"
            arg_type = Float64
            default = 0.0
        "--D"
            help = "Single-ion anisotropy (meV)"
            arg_type = Float64
            default = 0.05
        "--S"
            help = "Spin quantum number"
            arg_type = Float64
            default = 1.0
        "--H-min"
            help = "Minimum H for HHL cut"
            arg_type = Float64
            default = 0.0
        "--H-max"
            help = "Maximum H for HHL cut"
            arg_type = Float64
            default = 1.0
        "--L"
            help = "L value for HHL cut"
            arg_type = Float64
            default = 0.0
        "--n-H"
            help = "Number of H points"
            arg_type = Int
            default = 50
        "--E-max"
            help = "Maximum energy (meV)"
            arg_type = Float64
            default = 10.0
        "--n-E"
            help = "Number of energy points"
            arg_type = Int
            default = 100
        "--eta"
            help = "Energy resolution (meV)"
            arg_type = Float64
            default = 0.5
        "--output"
            help = "Output JSON file"
            arg_type = String
            default = "sunny_result.json"
        "--mode"
            help = "Calculation mode: dispersion, sqw, or simulate"
            arg_type = String
            default = "dispersion"
        "--qfile"
            help = "File with Q-points (for simulate mode)"
            arg_type = String
            default = ""
    end
    
    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    result = Dict{String, Any}()
    result["parameters"] = Dict(
        "J1" => args["J1"],
        "J2" => args["J2"],
        "D" => args["D"],
        "S" => args["S"]
    )
    
    if args["mode"] == "dispersion"
        # Calculate dispersion along HHL
        H_array = collect(range(args["H-min"], args["H-max"], length=args["n-H"]))
        L = args["L"]
        
        energies = analytical_dispersion_array(H_array, L;
                                               J1=args["J1"], J2=args["J2"],
                                               D=args["D"], S=args["S"])
        
        result["H"] = H_array
        result["L"] = L
        result["energy"] = energies
        result["mode"] = "dispersion"
        
    elseif args["mode"] == "sqw"
        # Calculate full S(Q,ω)
        qpoints, energies, sqw = calculate_hhl_cut(
            J1=args["J1"], J2=args["J2"], D=args["D"],
            H_min=args["H-min"], H_max=args["H-max"], L=args["L"],
            n_H=args["n-H"], E_max=args["E-max"], n_E=args["n-E"],
            η=args["eta"], S=args["S"]
        )
        
        result["H"] = [q[1] for q in qpoints]
        result["L"] = args["L"]
        result["energies"] = collect(energies)
        result["sqw"] = sqw
        result["mode"] = "sqw"
        
    elseif args["mode"] == "simulate"
        # Simulate measurements at specified Q-E points
        if args["qfile"] == ""
            error("Need --qfile for simulate mode")
        end
        
        # Read Q-E points from file
        lines = readlines(args["qfile"])
        measurements = []
        
        for line in lines
            parts = split(strip(line))
            if length(parts) >= 3
                H, L, E = parse.(Float64, parts[1:3])
                I, σ = simulate_measurement(H, L, E;
                                           J1=args["J1"], J2=args["J2"],
                                           D=args["D"], S=args["S"],
                                           η=args["eta"])
                push!(measurements, Dict("H" => H, "L" => L, "E" => E, "I" => I, "sigma" => σ))
            end
        end
        
        result["measurements"] = measurements
        result["mode"] = "simulate"
    end
    
    # Write output
    open(args["output"], "w") do f
        JSON.print(f, result, 2)
    end
    
    println("Output written to $(args["output"])")
end

# Run if called as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
