# Pyro によるガウス過程の実装

## ディレクトリ構成

* `/expamles`
    * 実行例
* `/models`
    * ガウス過程本体
* `/kernels`
    * カーネル関数
* `/likelihoods`
    * 尤度関数

## 実装したガウス過程の種類

* Variational Gaussian Process (変分ガウス過程) [1], [2], [3]
* Variational Sparse Gaussian Process (変分スパースガウス過程) [1], [2], [3]
* Stochastic Variational Sparse Gaussian Process (確率的変分スパースガウス過程) [1], [2], [3]
* Heteroscedastic Gaussian Process Regression (異分散ガウス過程回帰？) [4]
* Relevance Vector Machine (関連ベクトルマシン) [5]

詳細は[同時分布と変分事後分布の式](/model-and-guide.ipynb)を参照。

## 参考文献

* [1] 持橋大地, 大羽成征. "ガウス過程と機械学習". 講談社. 2019.
* [2] 須山敦志. "ベイズ深層学習". 講談社. 2019.
* [3] [Pyro公式の実装](https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/gp)
* [4] Miguel Lázaro-Gredilla, and Michalis K. Titsias. "Variational Heteroscedastic Gaussian Process Regression." ICML. 2011.
* [5] C.M.ビショップ. "パターン認識と機械学習". 丸善出版. 2012.
