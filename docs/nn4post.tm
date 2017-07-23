<TeXmacs|1.99.1>

<style|generic>

<\body>
  <section|Model>

  Suppose we have a model, <math|f<around*|(|x,\<theta\>|)>>, where <math|x>
  is the input and <math|\<theta\>> is the set of parameters of this model.
  Let <math|D> denotes an arbitrarily given dataset, i.e.
  <math|D=<around*|{|<around*|(|x<rsub|i>,y<rsub|i>|)>:i=1,2,\<ldots\>|}>>
  wherein for <math|\<forall\>i> <math|x<rsub|i>> is the input and
  <math|y<rsub|i>> the target (observed). With some assumption of the
  dataset, e.g. independency and Gaussianity, we can gain a likelihood
  <math|L<around*|(|D,\<theta\>|)>>. Suppose we have some prior on
  <math|\<theta\>>, <math|p<around*|(|\<theta\>|)>>, we gain the unormalized
  posterior <math|L<around*|(|D,\<theta\>|)> p<around*|(|\<theta\>|)>>. With
  <math|D> arbitrarily given, this unormalized posterior is a function of
  <math|\<theta\>>, denoted by <math|p<around*|(|\<theta\>;D|)>>.

  We we are going to do is fit this <math|p<rsub|D><around*|(|\<theta\>|)>>
  by ANN for any given <math|D>. To do so, we have to assume that
  <math|supp<around*|{|p<rsub|D><around*|(|\<theta\>|)>|}>=\<bbb-R\><rsup|m>>
  for some <math|m\<in\>\<bbb-N\><rsup|+>> (i.e. has no compact support) but
  decrease exponentially fast as <math|<around*|\<\|\|\>|\<theta\>|\<\|\|\>>\<rightarrow\>+\<infty\>>.
  With this assumption, we can use Gaussian function as the activation of the
  ANN. We propose the fitting function

  <\equation*>
    q<around*|(|\<theta\>|)>=<big|sum><rsub|i>a<rsub|i><rsup|2>
    <around*|{|<big|prod><rsub|j>N<around*|(|\<theta\><rsub|j>,w<rsub|j
    i>,b<rsub|j i>|)>|}>,
  </equation*>

  where <math|a<rsub|i>\<in\>\<bbb-R\>> for <math|\<forall\>i> (always
  <math|a<rsup|2><rsub|i>\<geqslant\>0>) and

  <\equation*>
    N<around*|(|x,w,b|)>\<assign\><sqrt|<frac|w<rsup|2>|2 \<pi\>>>
    exp<around*|(|-<frac|1|2><around*|(|w<rsup|2> x+b|)><rsup|2>|)>
  </equation*>

  While fitting, <math|q<around*|(|\<theta\>|)>> has no need of
  normalization, since <math|p<rsub|D><around*|(|\<theta\>|)>> is
  unormalized.

  <math|q<around*|(|\<theta\>|)>> has probablitic illustration.
  <math|N<around*|(|x,w,b|)>> is realized as a one-dimensional Gaussian
  distribution (denote <math|<with|math-font|cal|N>>). Indeed,
  <math|N<around*|(|x,w,b|)>=<with|math-font|cal|N<around*|(|x-\<mu\>,\<sigma\>|)>>>
  if <math|\<mu\>=b/w<rsup|2>> and <math|\<sigma\>=1/w<rsup|2>>. Thus
  <math|<big|prod><rsub|j> N<around*|(|\<theta\><rsub|j>,w<rsub|i j>,b<rsub|i
  j>|)>> is an multi-dimensional Gaussian distribution, with all dimensions
  independent. The <math|<around*|{|a<rsub|i><rsup|2>|}>> is an empirical
  distribution, randomly choosing the Gaussian distributions. Thus
  <math|q<around*|(|\<theta\>|)>> is a composition: <math|Empirical
  \<rightarrow\> Gaussian>. This is the <hlink|<em|mixture
  distribution>|https://en.wikipedia.org/wiki/Mixture_distribution>.

  Since there's no compact support, for both
  <math|p<rsub|D><around*|(|\<theta\>|)>> and
  <math|q<around*|(|\<theta\>|)>>, KL-divergence can be safely employed as
  the cost-function of the fitting.

  <section|Numerical Consideration>

  For numerical consideration, instead of fitting
  <math|p<rsub|D><around*|(|\<theta\>|)>> by <math|q<around*|(|\<theta\>|)>>,
  we fit <math|ln p<rsub|D><around*|(|\<theta\>|)>> by <math|ln
  q<around*|(|\<theta\>|)>>. To compute <math|ln q<around*|(|\<theta\>|)>>,
  we have to employ some approximation method. Let

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<beta\><rsub|i>>|<cell|\<assign\>>|<cell|ln<around*|(|a<rsub|i><rsup|2>
    <around*|{|<big|prod><rsub|j>N<around*|(|\<theta\><rsub|j>,w<rsub|j
    i>,b<rsub|j i>|)>|}>|)>>>|<row|<cell|>|<cell|=>|<cell|ln
    a<rsub|i><rsup|2>+<big|sum><rsub|j><around*|{|-<frac|1|2><around*|(|\<theta\><rsub|j>
    w<rsub|j i><rsup|2>+b<rsub|j i>|)><rsup|2>+<frac|1|2>ln<around*|(|<frac|w<rsub|j
    i><rsup|2>|2 \<pi\>>|)>|}>,>>>>
  </eqnarray*>

  thus <math|ln q<around*|(|\<theta\>|)>=ln<around*|(|<big|sum><rsub|i>exp<around*|(|\<beta\><rsub|i>|)>|)>.>
  We first compute all the <math|\<beta\><rsub|i>> and pick the maximum
  <math|\<beta\><rsub|max>>. Then pick out other <math|\<beta\><rsub|i>> for
  which <math|exp<around*|(|\<beta\><rsub|i>|)>> has the same order as
  <math|exp<around*|(|\<beta\><rsub|max>|)>>, collected as set <math|M>
  (excluding <math|\<beta\><rsub|max>>). We have

  <\eqnarray*>
    <tformat|<table|<row|<cell|ln q<around*|(|\<theta\>|)>>|<cell|=>|<cell|ln<around*|(|exp<around*|(|\<beta\><rsub|max>|)>+<big|sum><rsub|i\<neq\>max>exp<around*|(|\<beta\><rsub|i><rsup|>|)>|)>>>|<row|<cell|>|<cell|=>|<cell|ln<around*|(|1+<big|sum><rsub|i\<neq\>max>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>|)>+\<beta\><rsub|max>.>>>>
  </eqnarray*>

  Since <math|exp<around*|(|\<beta\><rsub|i>|)>\<sim\>exp<around*|(|\<beta\><rsub|max>|)>>
  iff <math|i\<in\>M> and <math|exp<around*|(|\<beta\><rsub|i>|)>\<ll\>exp<around*|(|\<beta\><rsub|max>|)>>
  iff <math|i\<nin\>M>, <math|<big|sum><rsub|i\<in\>M>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>\<sim\>1>,
  and others are negligible (comparing to <math|1>). Thus,

  <\equation*>
    ln q<around*|(|\<theta\>|)>\<approx\>ln<around*|(|1+<big|sum><rsub|i\<in\>M>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>|)>+\<beta\><rsub|max>.
  </equation*>

  This makes <math|ln q<around*|(|\<theta\>|)>> numerically computable. (And
  if <math|<big|sum><rsub|i\<in\>M>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>\<less\>0.1>,
  we can further approximate <math|ln q<around*|(|\<theta\>|)>\<approx\>\<beta\><rsub|max>+><math|<big|sum><rsub|i\<in\>M>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>>,
  thus no logrithm is to be computed.

  <section|Cost-Function>

  <\eqnarray*>
    <tformat|<table|<row|<cell|KL>|<cell|\<assign\>>|<cell|\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>-ln q<around*|(|\<theta\>|)>|]>>>|<row|<cell|>|<cell|\<approx\>>|<cell|<big|sum><rsub|\<theta\><rsup|<around*|(|s|)>><rsub|i>><around*|{|ln
    p<around*|(|\<theta\><rsub|i><rsup|<around*|(|s|)>>;D|)>-ln
    q<around*|(|\<theta\><rsub|i><rsup|<around*|(|s|)>>|)>|}>,>>>>
  </eqnarray*>

  where <math|\<theta\><rsup|<around*|(|s|)>>>s are sampled from
  <math|q<around*|(|\<theta\>|)>> as a distribution.
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
    <associate|auto-3|<tuple|3|?>>
  </collection>
</references>