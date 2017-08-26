using TopicModels

K = 3 # topic size
ja = Corpus("ja.txt")
en = Corpus("en.txt")
corpora = [ja, en]
beta_array = [0.1, 0.1]

model = PolylingualTM(corpora, beta_array, K)
train(model)

for l in 1:length(corpora)
    for topic in 1:K
        println()
        phi = word_predict(model, l, topic)
        for word_id in sortperm(phi, rev=true)
            print(get_word(corpora[l], word_id))
            @printf " %0.3f\n" phi[word_id]
        end
    end
end
