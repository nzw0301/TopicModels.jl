struct FPDLDA
    corpus::Corpus
    K::Int
    V::Int
    D::Int
    beta::Float64
    doc_dirichlet::Dirichlet
    nkv::Array{Int, 2} # cscmatrix
    ndk::Array{Int, 2} # cscmatrix
    nk::Array{Int, 1}
    z::Array{Array{Int, 1}, 1}

    function FPDLDA(corpus::Corpus, K=10, beta=0.01)
        @assert K > 0
        @assert beta > 0.

        topics = Array{Array{Int, 1}, 1}(corpus.D)
        nkv = zeros(Int, K, corpus.V)
        ndk = zeros(Int, corpus.D, K)
        nk = zeros(Int, K)

        for (doc_id, words) in enumerate(corpus.docs)
            z = rand(1:K, length(words))
            topics[doc_id] = z

            for (word, topic) in zip(words, z)
                nkv[topic, word] += 1
                ndk[doc_id, topic] += 1
                nk[topic] += 1
            end
        end

        new(corpus, K, corpus.V, corpus.D, beta,
            Dirichlet(K), nkv, ndk, nk, topics)
    end
end

function train(model::FPDLDA, iteration=777)
    function add(model::FPDLDA, doc_id::Int, word::Int, topic::Int, f_tree::FTree)
        model.nkv[topic, word] += 1
        model.ndk[doc_id, topic] += 1
        model.nk[topic] += 1
        add_update(f_tree, model.ndk[doc_id, topic] / (model.nk[topic] + model.beta) - get_node_value(f_tree, topic))
    end

    function remove(model::FPDLDA, doc_id::Int, word::Int, topic::Int, f_tree::FTree)
        model.nkv[topic, word] -= 1
        model.ndk[doc_id, topic] -= 1
        model.nk[topic] -= 1
        add_update(f_tree, model.ndk[doc_id, topic] / (model.nk[topic] + model.beta) - get_node_value(f_tree, topic))
    end

    function sample{Int}(model::FPDLDA, doc_id::Int, word::Int, f_tree::FTree)
        # init cumsum
        int2int = Dict{Int, Int}()
        pre_cumsum_term = 0.
        for k in 1:K
            if model.nkv[k, word] != 0.
                cum_sum[k] = pre_cumsum_term = pre_cumsum_term+(model.ndk[doc_id, k]+get_alpha(model.doc_dirichlet, k)) *
                     (model.nkv[k, word]+model.beta) / (model.beta*model.V+model.nk[k])
                 end

        end
        cum_sum = zeros(K)
        pre_cumsum_term = 0.

        u = rand() * (get_root_value(f_tree)) *
        K = model.K

        for k in 1:K
            cum_sum[k] = pre_cumsum_term = pre_cumsum_term+(model.ndk[doc_id, k]+get_alpha(model.doc_dirichlet, k)) *
                     (model.nkv[k, word]+model.beta) / (model.beta*model.V+model.nk[k])
        end

        return searchsortedfirst(cum_sum, rand()*pre_cumsum_term)
    end

    function update_start_ftree(model::FPDLDA, doc_id::Int, f_tree::FTree)
        for t in 1:model.K
            if model.ndk[doc_id, t] != 0
                add_update(t, model.ndk[doc_id, t])
            end
        end
    end

    function update_finish_ftree(model::FPDLDA, doc_id::Int, f_tree::FTree)
        for t in 1:model.K
            if model.ndk[doc_id, t] != 0
                add_update(t, -model.ndk[doc_id, t])
            end
        end
    end

    function init_doc_ftree(model::FPDLDA, doc_id::Int)
         Ftree(model.doc_dirichlet.get_alpha_all ./  (model.beta + model.nk))
    end

    for i in 1:iteration
        print("\r", i)
        doc_f_tree = init_doc_ftree(model, doc_id)
        for (doc_id, doc) in enumerate(model.corpus.docs)
            update_start_ftree(model, dic_id, doc_f_tree)

            for (z_id, w) in enumerate(doc)
                remove(model, doc_id, w, model.z[doc_id][z_id])

                z_dn = sample(model, doc_id, w, f_free)
                model.z[doc_id][z_id] = z_dn
                add(model, doc_id, w, z_dn)
            end
            update_finish_ftree(model, dic_id, doc_f_tree)
        end

    end
end

function word_predict(model::FPDLDA, topic_id::Int)
    return (model.nkv[topic_id, :] + model.beta) / (model.nk[topic_id] + model.V * model.beta)
end

function topic_predict(model::FPDLDA, doc_id::Int)
    p  = zeros(model.K)
    for k in 1:model.K
        p[k] = model.ndk[doc_id, k] + get_alpha(model.doc_dirichlet, k)
    end

    return p/sum(p)
end
