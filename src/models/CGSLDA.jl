struct CGSLDA
    corpus::Corpus
    K::Int
    V::Int
    D::Int
    beta::Float64
    doc_dirichlet::Dirichlet
    nkv::Array{Int, 2}
    ndk::Array{Int, 2}
    nk::Array{Int, 1}
    z::Array{Int, 1}

    function CGSLDA(corpus::Corpus, K=10, beta=0.01)
        @assert !corpus.is_row_document
        @assert K > 0
        @assert beta > 0.

        topics = zeros(Int, sum(corpus.docs))
        nkv = zeros(Int, K, corpus.V)
        ndk = zeros(Int, corpus.D, K)
        nk = zeros(Int, K)

        # init topics
        z_index = 1
        for doc_id in 1:corpus.D
            for i in nzrange(corpus.docs, doc_id)
                word_id = rowvals(corpus.docs)[i]
                for _ in 1:nonzeros(corpus.docs)[i]
                    topic = rand(1:K)
                    nkv[topic, word_id] += 1
                    ndk[doc_id, topic] += 1
                    nk[topic] += 1
                    topics[z_index] = topic
                    z_index += 1
                end
            end
        end
        new(corpus, K, corpus.V, corpus.D, beta,
            Dirichlet(K), nkv, ndk, nk, topics)
    end
end

function train(model::CGSLDA, iteration=777)
    function add(model::CGSLDA, doc_id::Int, word::Int, topic::Int)
        model.nkv[topic, word] += 1
        model.ndk[doc_id, topic] += 1
        model.nk[topic] += 1
    end

    function remove(model::CGSLDA, doc_id::Int, word::Int, topic::Int)
        model.nkv[topic, word] -= 1
        model.ndk[doc_id, topic] -= 1
        model.nk[topic] -= 1
    end

    function sample{Int}(model::CGSLDA, doc_id::Int, word::Int)
        K = model.K
        cum_sum = zeros(K)
        pre_cumsum_term = 0.
        for k in 1:K
            cum_sum[k] = pre_cumsum_term = pre_cumsum_term+(model.ndk[doc_id, k]+get_alpha(model.doc_dirichlet, k)) *
                     (model.nkv[k, word]+model.beta) / (model.beta*model.V+model.nk[k])
        end

        searchsortedfirst(cum_sum, rand()*pre_cumsum_term)
    end

    for i in 1:iteration
        z_index = 1
        print("\r", i)
        for doc_id in 1:model.corpus.D
            for i in nzrange(model.corpus.docs, doc_id)
                w = rowvals(model.corpus.docs)[i]
                for _ in 1:nonzeros(model.corpus.docs)[i]
                    remove(model, doc_id, w, model.z[z_index])
                    z_dn = sample(model, doc_id, w)
                    model.z[z_index] = z_dn
                    add(model, doc_id, w, z_dn)
                    z_index += 1
                end
            end
        end
    end
end

function word_predict(model::CGSLDA, topic_id::Int)
    (model.nkv[topic_id, :] + model.beta) / (model.nk[topic_id] + model.V * model.beta)
end

function topic_predict(model::CGSLDA, doc_id::Int)
    p  = zeros(model.K)
    for k in 1:model.K
        p[k] = model.ndk[doc_id, k] + get_alpha(model.doc_dirichlet, k)
    end

    p/sum(p)
end
