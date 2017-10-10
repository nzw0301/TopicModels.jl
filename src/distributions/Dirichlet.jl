struct Dirichlet
    k::Int
    alpha::Array{Float64, 1}
    sum_alpha::Float64

    function Dirichlet(alpha::Array{Float64, 1})
        new(length(alpha), alpha, sum(alpha))
    end
end

function Dirichlet(k::Int)
    @assert k > 0
    Dirichlet(fill(1/k, k))
end

function Dirichlet(k::Int, alpha::Float64)
    @assert k > 0
    @assert alpha > 0.0
    Dirichlet(fill(alpha, k))
end

function get_sum_alpha(dirichlet::Dirichlet)
    dirichlet.sum_alpha
end

function get_alpha(dirichlet::Dirichlet, i::Int)
    @assert 1 <= i <= dirichlet.k
    dirichlet.alpha[i]
end

function get_alpha_all(dirichlet::Dirichlet)
    dirichlet.alpha
end

