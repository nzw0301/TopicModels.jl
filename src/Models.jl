for fname in ["CGSLDA.jl", "PolylingualTM.jl", "FPDLDA.jl"]
    include(joinpath("models", fname))
end
