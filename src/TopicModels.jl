# __precompile__()

module TopicModels

export Corpus, get_document, get_word
export FTree, sample, add_update, get_node_value, get_root_value
export Dirichlet, get_alpha, get_sum_alpha, get_alpha_all
export CGSLDA, train, word_predict, topic_predict
export PolylingualTM, train, word_predict, topic_predict
export FPDLDA, train #, word_predict, topic_predict

include("Corpus.jl")
include("DataStructures.jl")
include("Distributions.jl")
include("Models.jl")

end
