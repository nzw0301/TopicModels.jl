@testset "Example case in F+LDA paper" begin
    ε = 1e-7
    data = [0.3, 1.5, 0.4, 0.3]
    tree = FTree(data)

    @test get_root_value(tree) ≈ sum(data) atol=ε
    @test get_node_value(tree, 3) == data[3]
    add_update!(tree, 3, 1.0)

    @test get_root_value(tree) ≈ sum(data)+1.0 atol=ε
    @test get_node_value(tree, 3) == data[3] + 1.0
end
