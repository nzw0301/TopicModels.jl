type CGSLDA
    corpus::Corpus
    k::Int64
    V::Int64
    D::Int64
    beta::Float64
    alpha::Array{Float64,1}
    nkv::Array{Int64,2}
    ndk::Array{Int64,2}
    nk::Array{Int64,1}
    z::Array{Array{Int64,1},1}

    function CGSLDA(corpus::Corpus, k=10, beta=0.01, alpha=0.1)
        topics = Array{Int64,1}[]
        nkv = zeros(Int64, k, corpus.V)
        ndk = zeros(Int64, corpus.D, k)
        nk = zeros(Int64, k)

        for (doc_id, words) in enumerate(corpus.docs)
            z = rand(1:k, length(words))
            push!(topics, z)

            for (word, topic) in zip(words, z)
                nkv[topic, word] += 1
                ndk[doc_id, topic] += 1
                nk[topic] += 1
            end
        end

        return new(corpus, k, corpus.V, corpus.D, beta,
                   fill(alpha, k), nkv, ndk, nk, topics)
    end
end

function train(model::CGSLDA, iteration=777)
    function add(model::CGSLDA, doc_id::Int64, word::Int64, topic::Int64)
        model.nkv[topic, word] += 1
        model.ndk[doc_id, topic] += 1
        model.nk[topic] += 1
    end

    function remove(model::CGSLDA, doc_id::Int64, word::Int64, topic::Int64)
        model.nkv[topic, word] -= 1
        model.ndk[doc_id, topic] -= 1
        model.nk[topic] -= 1
    end

    function sample{Int64}(model::CGSLDA, doc_id::Int64, word::Int64)
        k = model.k
        beta = model.beta
        V = model.V
        alpha = model.alpha
        pro = zeros(k)
        for t in 1:k
            pro[t] = (model.ndk[doc_id, t]+alpha[t])*(model.nkv[t, word]+beta) / (beta*V+model.nk[t])
        end

        u = rand()*sum(pro)
        for t in 1:k
            u -= pro[t]
            if u < 0.0
                return t
            end
        end
    end

    for i in 1:iteration
        print("\r", i)
        for (doc_id, doc) in enumerate(model.corpus.docs)
            for (z_id, w) in enumerate(doc)
                remove(model, doc_id, w, model.z[doc_id][z_id])
                z_dn = sample(model, doc_id, w)
                model.z[doc_id][z_id] = z_dn
                add(model, doc_id, w, z_dn)
            end
        end
    end
end

function word_predict(model::CGSLDA, topic_id::Int64)
    return (model.nkv[topic_id, :] + model.beta) / (model.nk[topic_id] + model.V * model.beta)
end

function topic_predict(model::CGSLDA, doc_id::Int64)
    p = model.ndk[doc_id, :] + model.alpha[model.k]
    return p/sum(p)
end
