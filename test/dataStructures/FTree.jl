@testset "Example case in F+LDA paper" begin
    ε = 1e-7
    data = [0.3, 1.5, 0.4, 0.3]
    tree = FTree(data)

    @test get_root_value(tree) ≈ sum(data) atol=ε
    @test get_node_value(tree, 3) == data[3]
    add_update!(tree, 3, 1.0)

    @test get_root_value(tree) ≈ sum(data)+1.0 atol=ε
    @test get_node_value(tree, 3) == data[3] + 1.0

    truth = data
    truth[3] += 1.0
    truth /= sum(truth)
    predicted = zeros(4)
    
    for _ in 1:1e7
        predicted[sample(tree)] += 1
    end

    @test truth ≈ predicted/sum(predicted) atol=1e-3
end
