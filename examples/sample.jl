using TopicModels

filename = "sample.txt"
K = 2
corpus = Corpus(filename)
lda = CGSLDA(corpus, K)
# lda = FPDLDA(corpus, K)
@time train(lda, 1000)

for topic in 1:K
    println()
    phi = word_predict(lda, topic)
    for word_id in sortperm(phi, rev=true)
       print(get_word(corpus, word_id))
       @printf " %0.3f\n" phi[word_id]
    end
end
