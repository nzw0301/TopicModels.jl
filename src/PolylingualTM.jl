struct PolylingualTM
    corpora::Array{Corpus, 1}
    K::Int
    Vl::Array{Int, 1}
    D::Int
    L::Int
    beta_array::Array{Float64, 1}
    doc_dirichlet::Dirichlet
    nlkv::Array{Array{Int, 2}, 1}
    nldk::Array{Int, 3}
    nlk::Array{Int, 2}
    z_array::Array{Array{Array{Int, 1}, 1}, 1}

    function PolylingualTM(corpora::Array{Corpus, 1}, beta_array::Array{Float64, 1}, K=10)
        @assert K > 0
        @assert length(corpora) == length(beta_array)

        num_lang = length(corpora)
        D = corpora[1].D
        Vl = [corpus.V for corpus in corpora]

        topics_array = Array{Array{Array{Int, 1}, 1}, 1}(num_lang)
        nlkv = Array{Array{Int, 2}, 1}(num_lang)
        nldk = zeros(Int, num_lang, D, K)
        nlk = zeros(Int, num_lang, K)

        for l in 1:num_lang
            topics = Array{Array{Int, 1}, 1}(D)
            nkv = zeros(Int, K, Vl[l])
            for (doc_id, words) in enumerate(corpora[l].docs)
                z = rand(1:K, length(words))
                topics[doc_id] = z

                for (word, topic) in zip(words, z)
                    nkv[topic, word] += 1
                    nldk[l, doc_id, topic] += 1
                    nlk[l, topic] += 1
                end
            end
            topics_array[l] = topics
            nlkv[l] = nkv
        end

        return new(corpora, K, Vl, D, num_lang, beta_array,
                   Dirichlet(K), nlkv, nldk, nlk, topics_array)
    end
end


function train(model::PolylingualTM, iteration=777)
    function add(model::PolylingualTM, lang::Int, doc_id::Int, word::Int, topic::Int)
        model.nlkv[lang][topic, word] += 1
        model.nldk[lang, doc_id, topic] += 1
        model.nlk[lang, topic] += 1
    end

    function remove(model::PolylingualTM, lang::Int, doc_id::Int, word::Int, topic::Int)
        model.nlkv[lang][topic, word] -= 1
        model.nldk[lang, doc_id, topic] -= 1
        model.nlk[lang, topic] -= 1
    end

    function sample{Int}(model::PolylingualTM, lang::Int, doc_id::Int, word::Int)
        K = model.K
        pro = zeros(K)
        for k in 1:K
            pro[k] = (model.nldk[lang, doc_id, k]+get_alpha(model.doc_dirichlet, k))*
                     (model.nlkv[lang][k, word]+model.beta_array[lang]) /
                     (model.beta_array[lang]*model.Vl[lang]+model.nlk[lang, k])
        end

        u = rand()*sum(pro)
        for k in 1:K
            u -= pro[k]
            if u < 0.0
                return k
            end
        end
    end

    for i in 1:iteration
        print("\r", i)
        for lang in 1:model.L
            for (doc_id, doc) in enumerate(model.corpora[lang].docs)
                for (z_id, w) in enumerate(doc)
                    remove(model, lang, doc_id, w, model.z_array[lang][doc_id][z_id])
                    z_ldn = sample(model, lang, doc_id, w)
                    model.z_array[lang][doc_id][z_id] = z_ldn
                    add(model, lang, doc_id, w, z_ldn)
                end
            end
        end
    end
end

function word_predict(model::PolylingualTM, lang_id::Int, topic_id::Int)
    @assert 0 <  lang_id <= model.L
    @assert 0 <  topic_id <= model.K

    return (model.nlkv[lang_id][topic_id, :] + model.beta_array[lang_id]) /
           (model.nlk[lang_id, topic_id] + model.Vl[lang_id] * model.beta_array[lang_id])
end

function topic_predict(model::PolylingualTM, doc_id::Int)
    @assert 0 <  doc_id <= model.D

    p  = zeros(model.k)
    for k in 1:model.K
        p[k] = sum(model.nldk[:, doc_id, K]) + get_alpha(model.doc_dirichlet, K)
    end

    return p/sum(p)
end
