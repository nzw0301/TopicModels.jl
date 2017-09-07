struct Corpus
    docs::SparseMatrixCSC{Int,Int}
    doc_lengths::Array{Int, 1}
    D::Int
    V::Int
    dictionary::Dictionary
    is_row_document::Bool

    function Corpus(path::String, is_row_document::Bool = false)
        dictionary = Dictionary()
        doc_lengths = Int[]
        doc_ids = Int[]
        word_ids = Int[]
        data = Int[]
        words_in_doc = Dict{Int, Int}()
        open(path) do f
            for (doc_id, line) in enumerate(readlines(f))
                for word in split(line)
                    word_id = update_and_get(dictionary, word)
                    words_in_doc[word_id] = get(words_in_doc, word_id, 0) + 1
                end

                doc_len = 0
                for (word_id, freq) in words_in_doc
                    push!(doc_ids, doc_id)
                    push!(word_ids, word_id)
                    push!(data, freq)
                    doc_len += 1
                end

                push!(doc_lengths, doc_len)
            end
        end

        if is_row_document
            docs = sparse(doc_ids, word_ids, data)
            D, V = size(docs)
        else
            docs = sparse(word_ids, doc_ids, data)
            V, D = size(docs)
        end
        new(docs, doc_lengths, D, V, dictionary, is_row_document)
    end
end
