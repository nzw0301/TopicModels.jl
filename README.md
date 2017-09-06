LinuxOS, OSX: [![Build Status](https://travis-ci.org/nzw0301/TopicModels.jl.svg)](https://travis-ci.org/nzw0301/TopicModels.jl)

Code Coverage: [![Coverage Status](https://coveralls.io/repos/nzw0301/topicModels.jl/badge.svg?branch=master)](https://coveralls.io/r/nzw0301/TopicModels.jl?branch=master) [![codecov.io](http://codecov.io/github/nzw0301/TopicModels.jl/coverage.svg?branch=master)](http://codecov.io/github/nzw0301/TopicMidels.jl?branch=master)



# TopicModels

## Implementations

- `CGSLDA` : LDA with collapsed Gibbs sampling. [Thomas L. Griffiths and Mark Steyvers. Finding Scientific Topics. PNAS, 2004.](http://psiexp.ss.uci.edu/research/papers/sciencetopics.pdf)
- `PolylingualTM` : Polylingual Topic Models with collapsed Gibbs sampling. [Polylingual Topic Models. David Mimno, Hanna M. Wallach, Jason Naradowsky, David A. Smith, Andrew McCallum. In Proc. EMNLP, 2009.](http://dirichlet.net/pdf/mimno09polylingual.pdf)
- `FPDLDA` : F+LDA (document-by-document): Topic scalable LDA with collapsed Gibbs sampling. [A Scalable Asynchronous Distributed Algorithm for Topic Modeling. Hsiang-Fu Yu, Cho-Jui Hsieh, Hyokun Yun, S.V.N Vishwanathan, Inderjit S. Dhillon. In Proc. WWW, 2015.](https://www.cs.utexas.edu/~rofuyu/papers/nomad-lda-www.pdf)
