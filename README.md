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

* [Variational Gaussian Process (変分ガウス過程)](#Variational-Gaussian-Process) [1], [2], [3]
* [Variational Sparse Gaussian Process (変分スパースガウス過程)](#Variational-Sparse-Gaussian-Process) [1], [2], [3]
* [Stochastic Variational Sparse Gaussian Process (確率的変分スパースガウス過程)](#Stochastic-Variational-Sparse-Gaussian-Process) [1], [2], [3]
* [Heteroscedastic Gaussian Process Regression (異分散ガウス過程回帰？)](#Heteroscedastic-Gaussian-Process-Regression) [4]

### Variational Gaussian Process

同時分布：

$$
p(\mathbf{y}, \mathbf{f} \mid \mathbf{X})
=
p(\mathbf{y} \mid \mathbf{f})
p(\mathbf{f} \mid \mathbf{X})
$$

$$
p(\mathbf{f} \mid \mathbf{X})
=
N(\mathbf{0}, k(\mathbf{X}, \mathbf{X}))
$$

変分事後分布：

$$
p(\mathbf{f} \mid \mathbf{X}, \mathbf{y})
\approx
q(\mathbf{f})
=
N(\mathbf{\mu}^*, \mathbf{\Sigma}^*)
$$

### Variational Sparse Gaussian Process

同時分布：

$$
p(\mathbf{y}, \mathbf{f}, \mathbf{u} \mid \mathbf{X})
=
p(\mathbf{y} \mid \mathbf{f})
p(\mathbf{f} \mid \mathbf{X}, \mathbf{u})
p(\mathbf{u} \mid \mathbf{Z})
$$

$$
p(\mathbf{f} \mid \mathbf{X}, \mathbf{u})
=
N(
    k(\mathbf{X}, \mathbf{Z})
    k(\mathbf{Z}, \mathbf{Z})^{-1}
    \mathbf{u},
    k(\mathbf{X}, \mathbf{X}) -
    k(\mathbf{X}, \mathbf{Z})
    k(\mathbf{Z}, \mathbf{Z})^{-1}
    k(\mathbf{Z}, \mathbf{X})
)
$$

$$
p(\mathbf{u} \mid \mathbf{Z})
=
N(\mathbf{0}, k(\mathbf{Z}, \mathbf{Z}))
$$

変分事後分布：

$$
p(\mathbf{f}, \mathbf{u} \mid \mathbf{X}, \mathbf{y})
\approx
p(\mathbf{f} \mid \mathbf{X}, \mathbf{u})
q(\mathbf{u})
$$

$$
q(\mathbf{u})
=
N(\mathbf{\mu}^*, \mathbf{\Sigma}^*)
$$

(疑問) 事後分布を $p(\mathbf{f}, \mathbf{u} \mid \mathbf{X}, \mathbf{y}) \approx p(\mathbf{f} \mid \mathbf{X}, \mathbf{u}) q(\mathbf{u})$ へ分解する近似は、大胆すぎないか？<br>
$p(\mathbf{f} \mid \mathbf{X}, \mathbf{y}, \mathbf{u}) = p(\mathbf{f} \mid \mathbf{X}, \mathbf{u})$ とは思えないし、そもそも $p(\mathbf{f} \mid \mathbf{X}, \mathbf{u})$ の式は $p(\mathbf{u} \mid \mathbf{Z}) = N(\mathbf{0}, k(\mathbf{Z}, \mathbf{Z}))$ という前提で導かれたものであり $q(\mathbf{u}) = N(\mathbf{\mu}^*, \mathbf{\Sigma}^*)$ へ更新された後では成り立たないのではないか？　

### Stochastic Variational Sparse Gaussian Process

同時分布：

$$
\begin{align*}
p(\mathbf{y}, \mathbf{f}, \mathbf{u} \mid \mathbf{X})
&=
p(\mathbf{y} \mid \mathbf{f})
p(\mathbf{f} \mid \mathbf{X}, \mathbf{u})
p(\mathbf{u} \mid \mathbf{Z}) \\
&=
\left\{\prod_{n}^{N}{
    p(y_n \mid f_n)
    p(f_n \mid \mathbf{x}_n, \mathbf{u})
}\right\}
p(\mathbf{u} \mid \mathbf{Z}) \\
&\approx
\left\{\prod_{m}^{M}{
    p(y_m \mid f_m)
    p(f_m \mid \mathbf{x}_m, \mathbf{u})
}\right\}^{\frac{N}{M}}
p(\mathbf{u} \mid \mathbf{Z})
\end{align*}
$$

$$
p(f_n \mid \mathbf{x}_n, \mathbf{u})
=
N(
    k(\mathbf{x}_n, \mathbf{Z})
    k(\mathbf{Z}, \mathbf{Z})^{-1}
    \mathbf{u},
    k(\mathbf{x}_n, \mathbf{x}_n) -
    k(\mathbf{x}_n, \mathbf{Z})
    k(\mathbf{Z}, \mathbf{Z})^{-1}
    k(\mathbf{Z}, \mathbf{x}_n)
)
$$

$$
p(\mathbf{u} \mid \mathbf{Z})
=
N(\mathbf{0}, k(\mathbf{Z}, \mathbf{Z}))
$$

変分事後分布：

$$
p(\mathbf{f}, \mathbf{u} \mid \mathbf{X}, \mathbf{y})
\approx
p(\mathbf{f} \mid \mathbf{X}, \mathbf{u})
q(\mathbf{u})
$$

$$
q(\mathbf{u})
=
N(\mathbf{\mu}^*, \mathbf{\Sigma}^*)
$$

### Heteroscedastic Gaussian Process Regression

同時分布：

$$
p(\mathbf{y}, \mathbf{f}, \mathbf{r} \mid \mathbf{X})
=
p(\mathbf{y} \mid \mathbf{f}, \mathbf{r})
p(\mathbf{f} \mid \mathbf{X})
p(\mathbf{r} \mid \mathbf{X})
$$

$$
p(\mathbf{y} \mid \mathbf{f}, \mathbf{r}) = N(\mathbf{f}, \mathbf{r})
$$

$$
p(\mathbf{f} \mid \mathbf{X})
=
N(\mathbf{0}, k_f(\mathbf{X}, \mathbf{X}))
$$

$$
p(\mathbf{r} \mid \mathbf{X})
=
N(\mathbf{\mu}_r, k_r(\mathbf{X}, \mathbf{X}))
$$

変分事後分布：

$$
p(\mathbf{f} \mid \mathbf{X}, \mathbf{y})
\approx
q(\mathbf{f})
=
N(\mathbf{\mu}^*_f, \mathbf{\Sigma}^*_f)
$$

$$
p(\mathbf{r} \mid \mathbf{X}, \mathbf{y})
\approx
q(\mathbf{r})
=
N(\mathbf{\mu}^*_r, \mathbf{\Sigma}^*_r)
$$

## 参考文献

* [1] 持橋大地, 大羽成征. "ガウス過程と機械学習". 講談社. 2019.
* [2] 須山敦志. "ベイズ深層学習". 講談社. 2019.
* [3] [Pyro公式の実装](https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/gp)
* [4] Miguel Lázaro-Gredilla, and Michalis K. Titsias. "Variational Heteroscedastic Gaussian Process Regression." ICML. 2011.
