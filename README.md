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

* Variational Gaussian Process (変分ガウス過程)
* Variational Sparse Gaussian Process (変分スパースガウス過程)
* Stochastic Variational Sparse Gaussian Process (確率的変分スパースガウス過程)

## 参考文献

* 持橋大地, 大羽成征. ガウス過程と機械学習. 講談社. 2019.
* 須山敦志. ベイズ深層学習. 講談社. 2019.
* [Pyro公式の実装](https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/gp)
