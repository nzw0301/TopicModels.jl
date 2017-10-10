struct FTree
    T::Int
    f::Array{Float64, 1}
    function FTree(p::Array{Float64, 1})
        function initTree(p::Array{Float64, 1}, T::Int)
            f = zeros(T*2-1)
            for i in (T-1)+length(p):-1:1
                f[i] = if T <= i
                    p[i-T+1]
                else
                    f[2i] + f[2i+1]
                end
            end
            return f
        end

        T = 2^Int(ceil(log2(length(p))))
        new(T, initTree(p, T))
    end
end

function add_update!(tree::FTree, t::Int, delta)
    @assert 1 <= t <= tree.T
    i = t + tree.T - 1
    while (i > 0)
        tree.f[i] += delta
        i = div(i, 2)
    end
end

function get_node_value(tree::FTree, t::Int)
    @assert 1 <= t <= tree.T
    tree.f[t+tree.T-1]
end

function get_root_value(tree::FTree)
    tree.f[1]
end

function sample(tree::FTree, u::Float64)
    i = 1

    while (i < tree.T)
        i = if u >= tree.f[2i]
            u -= tree.f[2i]
            2i+1
        else
            2i
        end
    end
    i - tree.T + 1
end

function sample(tree::FTree)
    u = rand() * get_root_value(tree)
    sample(tree, u)
end
