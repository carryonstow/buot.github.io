##Commodity Averaging Currency Swap

----

### Some Lognormal Relationships

If we let $A$ and $B$ be two lognormally distributed random variables, where we define
$$
\begin{align}
&\rm{E}[A] = \bar{A},  &&\rm{E}[B] = \bar{B}\\
&\rm{Var}[\rm{log}~A] = \mathit{v}_A, && \rm{Var}[\rm{log}~B] = \mathit{v}_B\\
&\rm{Cov}[\rm{log}~A, \rm{log}~B] = \mathit{v}_{AB}
\end{align}
$$
Now let's define a polynomial term
$$
C_{p,q} = A^p\cdot B^q
$$
for some constants $p, q$. Then because
$$
\rm{log}~C_{p,q} = p\cdot \rm{log}~A + q\cdot\rm{log}~B
$$
which is a linear combination of two normal distributions, i.e. another normal random  variable, with
$$
\rm{Var}[\rm{log}~C_{p,q}] = p^2 \mathit{v}_A + q^2\mathit{v}_B+2pq\mathit{v}_{AB}
$$
Meanwhile, we can infer that the expectation
$$
\displaystyle
\begin{align}
\rm{E}[A] = \bar{A}&=\rm{exp}~\left\{\rm{E}[\rm{log}~A]+\frac{1}{2}\rm{Var}~[\rm{log}~A]\right\}\\
&=\rm{exp}~\left\{\rm{E}[\rm{log}~A]+\frac{1}{2}\mathit{v}_A\right\}
\end{align}
$$
So we have
$$
\rm{E}[\rm{log}~A] = \rm{ln}\bar{A}-\frac{1}{2}\mathit{v}_A, \quad \rm{E}[\rm{log}~B] = \rm{ln}\bar{B}-\frac{1}{2}\mathit{v}_B
$$
Therefore
$$
\begin{align}
\mathrm{E}[\mathrm{log}~C_{p,q}] &= p\cdot \mathrm{ln}\bar{A} - \frac{1}{2}p\mathit{v}_A + q\cdot \mathrm{ln}\bar{B} - \frac{1}{2}q\mathit{v}_B\\
& = \mathrm{ln}\left(\bar{A}^p\bar{B}^q\right) - \frac{1}{2}\left(p\mathit{v}_A+q\mathit{v}_B\right)
\end{align}
$$
and we can derive the expectation of $C_{p,q}$ as
$$
\begin{align}
\mathrm{E}[C_{p,q}] &= \mathrm{exp}\left\{\mathrm{E}[\mathrm{log}~C_{p,q}] + \frac{1}{2}\mathrm{Var}[\mathrm{log}~C_{p,q}]\right\}\\
& = \mathrm{exp}\left\{\mathrm{ln}\left(\bar{A}^p\bar{B}^q\right) - \frac{1}{2}\left(p\mathit{v}_A+q\mathit{v}_B\right) + \frac{1}{2}p^2 \mathit{v}_A + q^2\mathit{v}_B+2pq\mathit{v}_{AB}\right\}\\
&=\bar{A}^p\bar{B}^q\mathrm{exp}\left\{\frac{1}{2}p(p-1)\mathit{v}_A + \frac{1}{2}q(q-1)\mathit{v}_B+2pq\mathit{v}_{AB}\right\}
\end{align}
$$
Similarly for its variance, using the definition of variance, it's readily seen that
$$
\begin{align}
\mathrm{Var}[C_{p,q}] &= \mathrm{E}[C_{p,q}^2] - \mathrm{E}^2[C_{p,q}]\\
&=  \mathrm{exp}\left\{2\mathrm{E}[\mathrm{log}~C_{p,q}] + 2\mathrm{Var}[\mathrm{log}~C_{p,q}]\right\}- \mathrm{exp}\left\{2\mathrm{E}[\mathrm{log}~C_{p,q}] + \mathrm{Var}[\mathrm{log}~C_{p,q}]\right\}\\
&=\mathrm{E}^2[C_{p,q}]~\left(\mathrm{exp}\left\{\mathrm{Var}[\mathrm{log}~C_{p,q}]\right\}-1\right)\\
&=\mathrm{E}^2[C_{p,q}]\left(\mathrm{exp}\left\{p^2 \mathit{v}_A + q^2\mathit{v}_B+2pq\mathit{v}_{AB}\right\} - 1\right)
\end{align}
$$
Now some special cases

- $C_{p,0} = A^p$
  $$
  \mathrm{E}[A^p]=\bar{A}^p~\mathrm{exp}\left\{\frac{1}{2}p(p-1)\mathit{v}_A\right\}\\
  \mathrm{Var}[\mathrm{log}~A^p] = p^2\mathit{v}_A
  $$

- $C_{1,1}=AB$
  $$
  \mathrm{E}[AB]=\bar{A}\bar{B}~\mathrm{exp}(\mathit{v}_{AB})\\
  \mathrm{Var}[\mathrm{log}~(AB)] = \mathit{v}_A + \mathit{v}_B + 2\mathit{v}_{AB}
  $$

- $C_{1,-1}=\frac{A}{B}$
  $$
  \mathrm{E}\left[\frac{A}{B}\right] = \frac{\bar{A}}{\bar{B}}\mathrm{exp}\left\{\mathit{v}_B-\mathit{v}_{AB}\right\}\\
  \mathrm{Var}\left[\mathrm{log}~\frac{A}{B}\right] = \mathit{v}_A+\mathit{v}_B-2\mathit{v}_{AB}
  $$


There is also a general formula for expectation of $Af(B)$ with $f$ some function, such that
$$
\mathrm{E}\left[Af(B)\right] = \mathrm{E}[A]\cdot\mathrm{E}[f(B\exp\{\mathit{v}_{AB}\})]
$$




