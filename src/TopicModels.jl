module TopicModels

export CGSLDA, train, word_predict, topic_predict
export Corpus, get_document, get_word

include("Corpus.jl")
include("CGSLDA.jl")

end
