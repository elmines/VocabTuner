
tiny_model =  [
              #Model options
              "--type", "s2s",
              "--dim-emb", str(128), "--dim-rnn", str(256),
              "--enc-cell", "gru", "--enc-cell-depth",      str(1), "--enc-depth", str(1),
              "--dec-cell", "gru", "--dec-cell-base-depth", str(1), "--dec-cell-high-depth", str(1), "--dec-depth", str(1),
              "--skip", "--layer-normalization", "--tied-embeddings",
              "--dropout-rnn", str(0.2), "--dropout-src", str(0.1), "--dropout-trg", str(0.1),
              #Training options
              "--max-length", str(50), "--max-length-crop",
              "--after-batches", str(1),
              "--device", str(0),
              "--mini-batch-fit",
              "--label-smoothing", str(0.1), "--exponential-smoothing"
              ]

big_custom =  [
              #General options
              "--workspace", str(8192), #8192MB = 8GB 
              #Model options
              "--type", "s2s",
              "--dim-emb", str(512), "--dim-rnn", str(1024),
              "--enc-cell", "lstm", "--enc-cell-depth",      str(2), "--enc-depth", str(4),
              "--dec-cell", "lstm", "--dec-cell-base-depth", str(4), "--dec-cell-high-depth", str(2), "--dec-depth", str(4),
              "--skip", "--layer-normalization", "--tied-embeddings",
              "--dropout-rnn", str(0.2), "--dropout-src", str(0.1), "--dropout-trg", str(0.1),
              #Training options
              "--max-length", str(50), "--max-length-crop",
              "--after-epochs", str(1),
              "--device", str(0),
              "--mini-batch", str(64),
              "--label-smoothing", str(0.1), "--exponential-smoothing"
              ]

best_deep =   [
              #General options
              "--workspace", str(8192), #8192MB = 8GB 
              #Model options
              "--type", "s2s",
              "--best-deep",
              #Training options
              "--max-length", str(50), "--max-length-crop",
              "--after-epochs", str(1),
              "--device", str(0),
              "--mini-batch", str(64)
              ]
