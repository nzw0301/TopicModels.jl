struct Dirichlet
    k::Int
    alpha::Array{Float64, 1}
    sum_alpha::Float64

    function Dirichlet(k::Int)
        new(k, ones(k)/k, 1.0)
    end
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
