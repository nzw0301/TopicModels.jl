struct FPDLDA
    corpus::Corpus
    K::Int
    V::Int
    D::Int
    beta::Float64
    doc_dirichlet::Dirichlet
    nkv::SparseMatrixCSC{Int,Int}
    nkd::SparseMatrixCSC{Int,Int}
    nk::Array{Int, 1}
    z::Array{Array{Int, 1}, 1}

    function FPDLDA(corpus::Corpus, K=10, beta=0.01)
        @assert K > 0
        @assert beta > 0.

        topics = Array{Array{Int, 1}, 1}(corpus.D)
        nkv = spzeros(Int, K, corpus.V)
        nkd = spzeros(Int, K, corpus.D)
        nk = zeros(Int, K)

        # init topic
        for (doc_id, words) in enumerate(corpus.docs)
            z = rand(1:K, length(words))
            topics[doc_id] = z

            for (word, topic) in zip(words, z)
                nkv[topic, word] += 1
                nkd[topic, doc_id] += 1
                nk[topic] += 1
            end
        end

        new(corpus, K, corpus.V, corpus.D, beta,
            Dirichlet(K), nkv, nkd, nk, topics)
    end
end

function train(model::FPDLDA, iteration=777)
    function add(model::FPDLDA, doc_id::Int, word::Int, topic::Int, f_tree::FTree)
        model.nkv[topic, word] += 1
        model.nkd[topic, doc_id] += 1
        model.nk[topic] += 1

        add_update(f_tree,
                   topic,
                   (model.nkd[topic, doc_id] + get_alpha(model.doc_dirichlet, topic))
                    / (model.nk[topic] + model.V * model.beta) - get_node_value(f_tree, topic))
    end

    function remove(model::FPDLDA, doc_id::Int, word::Int, topic::Int, f_tree::FTree)
        model.nkv[topic, word] -= 1
        model.nkd[topic, doc_id] -= 1
        model.nk[topic] -= 1

        add_update(f_tree,
                   topic,
                   (model.nkd[topic, doc_id] + get_alpha(model.doc_dirichlet, topic))
                    / (model.nk[topic] + model.V * model.beta) - get_node_value(f_tree, topic))
    end

    function sample{Int}(model::FPDLDA, word::Int, f_tree::FTree)
        topic_ids = rowvals(model.nkv)[nzrange(model.nkv, word)]
        r = zeros(length(topic_ids))
        pre_cumsum_term = 0.
        for (i, k) in enumerate(topic_ids)
            r[i] = pre_cumsum_term = pre_cumsum_term + model.nkv[k, word]*get_node_value(f_tree, k)
        end

        u = rand() * (get_root_value(f_tree) + r[end])
        if u <= r[end]
            return topic_ids[searchsortedfirst(r, u)]
        else
            return discrete(f_tree, (u-r[end])/model.beta)
        end
    end

    function update_start_ftree(model::FPDLDA, doc_id::Int, f_tree::FTree)
        topic_ids = rowvals(model.nkd)[nzrange(model.nkd, doc_id)]
        for t in topic_ids
            add_update(f_tree, t, model.nkd[t, doc_id] / (model.V * model.beta + model.nk[t]))
        end
    end

    function update_finish_ftree(model::FPDLDA, doc_id::Int, f_tree::FTree)
        topic_ids = rowvals(model.nkd)[nzrange(model.nkd, doc_id)]
        for t in topic_ids
            add_update(f_tree, t, -model.nkd[t, doc_id] / (model.V * model.beta + model.nk[t]))
        end
    end

    function init_doc_ftree(model::FPDLDA)
         return FTree(get_alpha_all(model.doc_dirichlet) ./  (model.V * model.beta + model.nk))
    end

    for i in 1:iteration
        print("\r", i)
        doc_f_tree = init_doc_ftree(model)
        for (doc_id, doc) in enumerate(model.corpus.docs)
            update_start_ftree(model, doc_id, doc_f_tree)

            for (z_id, w) in enumerate(doc)
                remove(model, doc_id, w, model.z[doc_id][z_id], doc_f_tree)

                z_dn = sample(model, w, doc_f_tree)
                model.z[doc_id][z_id] = z_dn
                add(model, doc_id, w, z_dn, doc_f_tree)
            end
            update_finish_ftree(model, doc_id, doc_f_tree)
        end
    end
end

function word_predict(model::FPDLDA, topic_id::Int)
    return (model.nkv[topic_id, :] + model.beta) / (model.nk[topic_id] + model.V * model.beta)
end

function topic_predict(model::FPDLDA, doc_id::Int)
    p  = zeros(model.K)
    for k in 1:model.K
        p[k] = model.nkd[k, doc_id] + get_alpha(model.doc_dirichlet, k)
    end

    return p/sum(p)
end
