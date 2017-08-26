struct Corpus
    docs::Array{Array{Int, 1}, 1}
    doc_length::Array{Int, 1}
    w2i::Dict{String, Int}
    i2w::Array{String, 1}
    V::Int
    D::Int

    function Corpus(path::String)
        w2i = Dict{String, Int}()
        i2w = String[]
        doc_lengths = Int[]
        docs = Array{Int, 1}[]
        open(path) do f
            for line in readlines(f)
                doc = split(line)
                word_ids = Int[]
                for w in doc
                    w_id = get(w2i, w, length(w2i)+1)
                    push!(word_ids, w_id)
                    if length(w2i) < w_id # new
                        w2i[w] = w_id
                        push!(i2w, w)
                    end
                end
                push!(docs, word_ids)
                push!(doc_lengths, length(word_ids))
            end
        end

        return new(docs, doc_lengths, w2i, i2w, length(w2i), length(docs))
    end
end

function get_word(corpus::Corpus, w_id::Int)
    return corpus.i2w[w_id]
end

function get_document(corpus::Corpus, doc_id::Int)
    return corpus.docs[doc_id]
end
