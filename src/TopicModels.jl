module TopicModels

export CGSLDA, train, word_predict, topic_predict
export Corpus, get_document, get_word
export Dirichlet, get_alpha, get_sum_alpha

include("Corpus.jl")
include("CGSLDA.jl")
include("Distributions.jl")

end
