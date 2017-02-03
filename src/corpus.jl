type Corpus
    docs::Array{Array{Int64,1},1}
    doc_length::Array{Int64,1}
    w2i::Dict{String, Int64}
    i2w::Array{String,1}
    V::Int64
    D::Int64

    function Corpus(path::String)
        w2i = Dict{String, Int64}()
        i2w = String[]
        doc_lenths = Int64[]
        docsIds = Array{Int64,1}[]
        open(path) do f
            for line in readlines(f)
                doc = split(line)
                wordIDs = Int64[]
                for w in doc
                    w_id = get(w2i, w, length(w2i)+1)
                    push!(wordIDs, w_id)
                    if length(w2i) < w_id # new
                        w2i[w] = w_id
                        push!(i2w, w)
                    end
                end
                push!(docsIds, wordIDs)
                push!(doc_lenths, length(wordIDs))
            end
        end

        return new(docsIds, doc_lenths, w2i, i2w, length(w2i), length(docsIds))
    end
end

function get_word(corpus::Corpus, w_id::Int64)
    return corpus.i2w[w_id]
end

function get_document(corpus::Corpus, doc_id::Int64)
    return corpus.docs[doc_id]
end
