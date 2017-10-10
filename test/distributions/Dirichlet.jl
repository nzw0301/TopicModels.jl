@testset "Simple tests" begin
    ε = 1e-7
    @testset "Int initialization" begin
        k = 5
        dist = Dirichlet(k)
        @test get_alpha(dist, 1) ≈ 1/k atol=ε
        @test get_sum_alpha(dist) ≈ 1.0 atol=ε
        @test get_alpha_all(dist) ≈ ones(k)/k atol=ε
    end

    @testset "symmetric float initialization" begin
        a = 0.1
        k = 5
        dist = Dirichlet(k, a)
        @test get_alpha(dist, 1) ≈ a atol=ε
        @test get_sum_alpha(dist) ≈ a*k atol=ε
        @test get_alpha_all(dist) ≈ ones(k)*a atol=ε
    end

    @testset "asymmetric float array initialization" begin
        alpha = [1.0, 2.0, 1.0]
        dist = Dirichlet(alpha)
        @test get_alpha(dist, 1) ≈ alpha[1] atol=ε
        @test get_sum_alpha(dist) ≈ sum(alpha) atol=ε
        @test get_alpha_all(dist) ≈ alpha atol=ε
        @test dist.k == 3
    end
end
