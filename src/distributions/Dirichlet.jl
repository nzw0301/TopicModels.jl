struct  Dirichlet
    k::Int
    alpha::Array{Float64, 1}
    sum_alpha::Float64

    function Dirichlet(k::Int)
            new(k, ones(k)/k, 1.0)
    end
end

function get_sum_alpha(dirichlet::Dirichlet)
    return dirichlet.sum_alpha
end

function get_alpha(dirichlet::Dirichlet, i::Int)
    @assert 1 <= i <= dirichlet.k
    return dirichlet.alpha[i]
end
