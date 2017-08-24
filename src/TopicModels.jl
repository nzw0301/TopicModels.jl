module TopicModels

export Corpus, get_document, get_word
export Dirichlet, get_alpha, get_sum_alpha
export CGSLDA, train, word_predict, topic_predict

include("Corpus.jl")
include("Distributions.jl")
include("CGSLDA.jl")
include("PolylingualLDA.jl")

end
