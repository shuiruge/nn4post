<TeXmacs|1.99.1>

<style|generic>

<\body>
  <section|Preliminary>

  <subsection|Assumptions on Posterior>

  Let <math|f<around*|(|x;\<theta\>|)>> a function of <math|x> with parameter
  <math|\<theta\>>. Let <math|y=f<around*|(|x;\<theta\>|)>> an observable,
  thus the observed value obeys a Gaussian distribution. Thus, for a list of
  observations <math|D\<assign\><around*|{|<around*|(|x<rsub|i>,y<rsub|i>,\<sigma\><rsub|i>|)>:i=1,\<ldots\>,n|}>>
  (<math|\<sigma\><rsub|i>> is the observational error of <math|y<rsub|i>>),
  we can construct a (logrithmic) likelihood, as

  <\eqnarray*>
    <tformat|<table|<row|<cell|ln p<around*|(|D\|\<theta\>|)>>|<cell|=>|<cell|ln<around*|(|<big|prod><rsub|i=1><rsup|n><frac|1|<sqrt|2
    \<pi\> \<sigma\><rsub|i><rsup|2>>> exp<around*|{|-<frac|1|2>
    <around*|(|<frac|y<rsub|i>-f<around*|(|x<rsub|i>;\<theta\>|)>|\<sigma\><rsub|i>>|)><rsup|2>|}>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|n><around*|{|-<frac|1|2>ln
    <around*|(|2 \<pi\> \<sigma\><rsub|i><rsup|2>|)>-<frac|1|2>
    <around*|(|<frac|y<rsub|i>-f<around*|(|x<rsub|i>;\<theta\>|)>|\<sigma\><rsub|i>>|)><rsup|2>|}>.>>>>
  </eqnarray*>

  If in addition assume a Gaussian prior, for some hyper-parameter
  <math|\<sigma\>>,

  <\equation*>
    p<around*|(|\<theta\>|)>=<frac|1|<sqrt|2 \<pi\> \<sigma\><rsup|2>>>
    exp<around*|(|-<frac|\<theta\><rsup|2>|2 \<sigma\><rsup|2>>|)>,
  </equation*>

  then we have posterior <math|p<around*|(|\<theta\>\|D|)>>

  <\eqnarray*>
    <tformat|<table|<row|<cell|ln p<around*|(|\<theta\>\|D|)>>|<cell|=>|<cell|-<frac|1|2><around*|{|<big|sum><rsub|i=1><rsup|n>
    <around*|(|<frac|y<rsub|i>-f<around*|(|x<rsub|i>;\<theta\>|)>|\<sigma\><rsub|i>>|)><rsup|2>+<around*|(|<frac|\<theta\>|\<sigma\>>|)><rsup|2>|}>>>|<row|<cell|>|<cell|->|<cell|<frac|1|2><around*|{|<big|sum><rsub|i=1><rsup|n>
    ln <around*|(|2 \<pi\> \<sigma\><rsub|i><rsup|2>|)>+ln <around*|(|2
    \<pi\> \<sigma\><rsup|2>|)>|}>,>>>>
  </eqnarray*>

  where the second line is <math|\<theta\>>-independent.

  <subsection|Bayesian Inference>

  Sample <math|m> samples from <math|p<around*|(|\<theta\>\|D|)>>,
  <math|<around*|{|\<theta\><rsub|<around*|(|s|)>>:s=1,\<ldots\>,m|}>>. Thus,
  the Bayesian inference gives prediction from <math|x> to <math|y> as

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|y|^>>|<cell|=>|<cell|\<bbb-E\><rsub|\<theta\>\<sim\>p<around*|(|\<theta\>\|D|)>><around*|[|f<around*|(|x;\<theta\>|)>|]>>>|<row|<cell|>|<cell|\<approx\>>|<cell|<around*|(|<frac|1|m><big|sum><rsub|s=1><rsup|m>|)>f<around*|(|x;\<theta\><rsub|<around*|(|s|)>>|)>.>>>>
  </eqnarray*>

  <section|Neural Network for Posterior>

  <subsection|Model>

  Suppose we have a model, <math|f<around*|(|x,\<theta\>|)>>, where <math|x>
  is the input and <math|\<theta\>> is the set of parameters of this model.
  Let <math|D> denotes an arbitrarily given dataset, i.e.
  <math|D=<around*|{|<around*|(|x<rsub|i>,y<rsub|i>|)>:i=1,2,\<ldots\>|}>>
  wherein for <math|\<forall\>i> <math|x<rsub|i>> is the input and
  <math|y<rsub|i>> the target (observed). With some assumption of the
  dataset, e.g. independency and Gaussianity, we can gain a likelihood
  <math|L<around*|(|\<theta\>;D|)>\<assign\>p<around*|(|D\|\<theta\>|)>>.
  Suppose we have some prior on <math|\<theta\>>,
  <math|p<around*|(|\<theta\>|)>>, we gain the unormalized posterior
  <math|L<around*|(|D,\<theta\>|)> p<around*|(|\<theta\>|)>>. With <math|D>
  arbitrarily given, this unormalized posterior is a function of
  <math|\<theta\>>, denoted by <math|p<around*|(|\<theta\>;D|)>>.

  We we are going to do is fit this <math|p<rsub|D><around*|(|\<theta\>|)>>
  by ANN for any given <math|D>. To do so, we have to assume that
  <math|supp<around*|{|p<around*|(|\<theta\>;D|)>|}>=\<bbb-R\><rsup|m>> for
  some <math|m\<in\>\<bbb-N\><rsup|+>> (i.e. has no compact support) but
  decrease exponentially fast as <math|<around*|\<\|\|\>|\<theta\>|\<\|\|\>>\<rightarrow\>+\<infty\>>.
  With this assumption, we can use Gaussian function as the activation of the
  ANN. We propose the fitting function

  <\equation*>
    q<around*|(|\<theta\>;a,b,w|)>=<big|sum><rsub|i>a<rsub|i><rsup|2>
    <around*|{|<big|prod><rsub|j>N<around*|(|\<theta\><rsub|j>,w<rsub|j
    i>,b<rsub|j i>|)>|}>,
  </equation*>

  where <math|a<rsub|i>\<in\>\<bbb-R\>> for <math|\<forall\>i> (always
  <math|a<rsup|2><rsub|i>\<geqslant\>0>) and

  <\equation*>
    N<around*|(|x,w,b|)>\<assign\><sqrt|<frac|w<rsup|2>|2 \<pi\>>>
    exp<around*|(|-<frac|1|2><around*|(|w x+b|)><rsup|2>|)>
  </equation*>

  While fitting, <math|q<around*|(|\<theta\>;a,b,w|)>> has no need of
  normalization, since <math|p<around*|(|\<theta\>;D|)>> is unormalized.

  <math|q<around*|(|\<theta\>;a,b,w|)>> has probablitic illustration.
  <math|N<around*|(|x,w,b|)>> is realized as a one-dimensional Gaussian
  distribution (denote <math|<with|math-font|cal|N>>). Indeed,
  <math|N<around*|(|x,w,b|)>=<with|math-font|cal|N<around*|(|x-\<mu\>,\<sigma\>|)>>>
  if <math|\<mu\>=b/w> and <math|\<sigma\>=1/<around*|\||w|\|>>. Thus
  <math|<big|prod><rsub|j> N<around*|(|\<theta\><rsub|j>,w<rsub|i j>,b<rsub|i
  j>|)>> is an multi-dimensional Gaussian distribution, with all dimensions
  independent. The <math|<around*|{|a<rsub|i><rsup|2>|}>> is an empirical
  distribution, randomly choosing the Gaussian distributions. Thus
  <math|q<around*|(|\<theta\>|)>> is a composition: <math|Empirical
  \<rightarrow\> Gaussian>. This is the <hlink|<em|mixture
  distribution>|https://en.wikipedia.org/wiki/Mixture_distribution>.

  Since there's no compact support, for both
  <math|p<around*|(|\<theta\>;D|)>> and <math|q<around*|(|\<theta\>;a,b,w|)>>,
  KL-divergence (equivalently, ELBO) can be safely employed as the
  cost-function of the fitting.

  <subsection|Numerical Consideration>

  For numerical consideration, instead of fitting
  <math|p<around*|(|\<theta\>;D|)>> by <math|q<around*|(|\<theta\>;a,b,w|)>>,
  we fit <math|ln p<around*|(|\<theta\>;D|)>> by <math|ln
  q<around*|(|\<theta\>;a,b,w|)>>. To compute <math|ln
  q<around*|(|\<theta\>;a,b,w|)>>, we have to employ some approximation
  method. Let

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<beta\><rsub|i>>|<cell|\<assign\>>|<cell|ln<around*|(|a<rsub|i><rsup|2>
    <around*|{|<big|prod><rsub|j>N<around*|(|\<theta\><rsub|j>,w<rsub|j
    i>,b<rsub|j i>|)>|}>|)>>>|<row|<cell|>|<cell|=>|<cell|ln
    a<rsub|i><rsup|2>+<big|sum><rsub|j><around*|{|-<frac|1|2><around*|(|\<theta\><rsub|j>
    w<rsub|j i>+b<rsub|j i>|)><rsup|2>+<frac|1|2>ln<around*|(|<frac|w<rsub|j
    i><rsup|2>|2 \<pi\>>|)>|}>,>>>>
  </eqnarray*>

  thus <math|ln q=ln<around*|(|<big|sum><rsub|i>exp<around*|(|\<beta\><rsub|i>|)>|)>.>
  We first compute all the <math|\<beta\><rsub|i>> and pick the maximum
  <math|\<beta\><rsub|max>>. Then pick out other <math|\<beta\><rsub|i>> for
  which <math|exp<around*|(|\<beta\><rsub|i>|)>> has the same order as
  <math|exp<around*|(|\<beta\><rsub|max>|)>>, collected as set <math|M>
  (excluding <math|\<beta\><rsub|max>>). We have

  <\eqnarray*>
    <tformat|<table|<row|<cell|ln q>|<cell|=>|<cell|ln<around*|(|exp<around*|(|\<beta\><rsub|max>|)>+<big|sum><rsub|i\<neq\>max>exp<around*|(|\<beta\><rsub|i><rsup|>|)>|)>>>|<row|<cell|>|<cell|=>|<cell|ln<around*|(|1+<big|sum><rsub|i\<neq\>max>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>|)>+\<beta\><rsub|max>.>>>>
  </eqnarray*>

  Since <math|exp<around*|(|\<beta\><rsub|i>|)>\<sim\>exp<around*|(|\<beta\><rsub|max>|)>>
  iff <math|i\<in\>M<rsub|\<beta\>>> and <math|exp<around*|(|\<beta\><rsub|i>|)>\<ll\>exp<around*|(|\<beta\><rsub|max>|)>>
  iff <math|i\<nin\>M<rsub|\<beta\>>>, <math|<big|sum><rsub|i\<in\>M<rsub|\<beta\>>>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>\<sim\>1>,
  and others are negligible (comparing to <math|1>). Thus,

  <\equation*>
    ln q\<approx\>ln<around*|(|1+<big|sum><rsub|i\<in\>M<rsub|\<beta\>>>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>|)>+\<beta\><rsub|max>.
  </equation*>

  This makes <math|ln q> numerically computable. (And if
  <math|<big|sum><rsub|i\<in\>M<rsub|\<beta\>>>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>\<less\>0.1>,
  we can further approximate <math|ln q\<approx\>\<beta\><rsub|max>+><math|<big|sum><rsub|i\<in\>M<rsub|\<beta\>>>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>>,
  thus no logrithm is to be computed.

  <subsection|Cost-Function (Performance)>

  <\eqnarray*>
    <tformat|<table|<row|<cell|ELBO<around*|(|a,b,w|)>>|<cell|\<assign\>>|<cell|\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>;w,b|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>-ln q<around*|(|\<theta\>;a,b,w|)>|]>>>|<row|<cell|>|<cell|\<approx\>>|<cell|<around*|(|<frac|1|n>
    <big|sum><rsub|\<theta\><rsup|<around*|(|s|)>>>|)><around*|{|ln
    p<around*|(|\<theta\><rsub|<around*|(|s|)>>;D|)>-ln
    q<around*|(|\<theta\><rsub|<around*|(|s|)>>;a,b,w|)>|}>,>>>>
  </eqnarray*>

  where <math|<around*|{|\<theta\><rsub|<around*|(|s|)>>: s=1,\<ldots\>,n|}>>
  is sampled from <math|q<around*|(|\<theta\>;a,b,w|)>> as a distribution.

  <subsection|Gradient>

  Let <math|z\<assign\><around*|(|a,b,w|)>>. Then,

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>ELBO|\<partial\>z><around*|(|z|)>>|<cell|=>|<cell|<frac|\<partial\>|\<partial\>z><big|int>\<mathd\>\<theta\>
    q<around*|(|\<theta\>;z|)> <around*|{|ln p<around*|(|\<theta\>;D|)>-ln
    q<around*|(|\<theta\>;z|)>|}>>>|<row|<cell|>|<cell|=>|<cell|<big|int>\<mathd\>\<theta\>
    q<around*|(|\<theta\>;z|)> <frac|\<partial\>ln
    q|\<partial\>z><around*|(|\<theta\>;z|)> <around*|{|ln
    p<around*|(|\<theta\>;D|)>-ln q<around*|(|\<theta\>;z|)>-1|}>>>|<row|<cell|>|<cell|\<approx\>>|<cell|<frac|1|n>
    <big|sum><rsub|\<theta\><rsub|<around*|(|s|)>>><frac|\<partial\>ln
    q|\<partial\>z><around*|(|\<theta\><rsub|<around*|(|s|)>>;z|)>
    <around*|{|ln p<around*|(|\<theta\><rsub|<around*|(|s|)>>;D|)>-ln
    q<around*|(|\<theta\><rsub|<around*|(|s|)>>;z|)>-1|}>>>>>
  </eqnarray*>

  where <math|<around*|{|\<theta\><rsub|<around*|(|s|)>>: s=1,\<ldots\>,n|}>>
  is sampled from <math|q<around*|(|\<theta\>;z|)>> as a distribution. Next,
  since <math|ln q=ln<around*|(|<big|sum><rsub|i>exp<around*|(|\<beta\><rsub|i>|)>|)>>,
  we have

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>ln
    q|\<partial\>z><around*|(|\<theta\>;z|)>>|<cell|=>|<cell|<big|sum><rsub|i><frac|exp<around*|(|\<beta\><rsub|i>|)>|<big|sum><rsub|j>exp<around*|(|\<beta\><rsub|j>|)>>
    <frac|\<partial\>\<beta\><rsub|i>|\<partial\>z>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i><frac|exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>|<big|sum><rsub|j>exp<around*|(|\<beta\><rsub|j>-\<beta\><rsub|max>|)>>
    <frac|\<partial\>\<beta\><rsub|i>|\<partial\>z>.>>>>
  </eqnarray*>

  Since <math|\<partial\>\<beta\><rsub|i>/\<partial\>z> is polynormial-like,
  thus

  <\equation*>
    <frac|\<partial\>ln q|\<partial\>z><around*|(|\<theta\>;z|)>\<approx\><big|sum><rsub|i><frac|exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>|<big|sum><rsub|j\<in\>M<rsub|\<beta\>>>exp<around*|(|\<beta\><rsub|j>-\<beta\><rsub|max>|)>>
    <frac|\<partial\>\<beta\><rsub|i>|\<partial\>z>
    \<delta\><rsub|i\<in\>M<rsub|\<beta\>>>,
  </equation*>

  where <math|M<rsub|\<beta\>>> is defined as previous. To calculate
  <math|\<partial\>\<beta\><rsub|i>/\<partial\>a<rsub|k>>,
  <math|\<partial\>\<beta\><rsub|i>/\<partial\>b<rsub|j k>> and
  <with|font-series|bold|<math|\<partial\>\<beta\><rsub|i>/\<partial\>w<rsub|j
  k>>>, recall

  <\equation*>
    \<beta\><rsub|i>=ln a<rsub|i><rsup|2>+<big|sum><rsub|j><around*|{|-<frac|1|2><around*|(|\<theta\><rsub|j>
    w<rsub|j i>+b<rsub|j i>|)><rsup|2>+<frac|1|2>ln<around*|(|<frac|w<rsub|j
    i><rsup|2>|2 \<pi\>>|)>|}>,
  </equation*>

  we thus have

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>\<beta\><rsub|i>|\<partial\>a<rsub|k>>>|<cell|=>|<cell|\<delta\><rsub|i
    k> <frac|2|a<rsub|k>>;>>|<row|<cell|<frac|\<partial\>\<beta\><rsub|i>|\<partial\>b<rsub|j
    k>>>|<cell|=>|<cell|-\<delta\><rsub|i k> <around*|{|\<theta\><rsub|j>
    w<rsub|j k>+b<rsub|j k>|}>;>>|<row|<cell|<frac|\<partial\>\<beta\><rsub|i>|\<partial\>w<rsub|j
    k>>>|<cell|=>|<cell|-\<delta\><rsub|i k>
    <around*|{|<around*|(|\<theta\><rsub|j> w<rsub|j k>+b<rsub|j k>|)>
    \<theta\><rsub|j>+<frac|1|w<rsub|j k>>|}>.>>>>
  </eqnarray*>

  And recall

  <\equation*>
    <frac|\<partial\>ELBO|\<partial\>z><around*|(|z|)>\<approx\><around*|(|<frac|1|n>
    <big|sum><rsub|\<theta\><rsub|<around*|(|s|)>>>|)> <around*|{|ln
    p<around*|(|\<theta\><rsub|<around*|(|s|)>>;D|)>-ln
    q<around*|(|\<theta\><rsub|<around*|(|s|)>>;z|)>-1|}>
    <big|sum><rsub|i><frac|exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>|<big|sum><rsub|j\<in\>M<rsub|\<beta\>>>exp<around*|(|\<beta\><rsub|j>-\<beta\><rsub|max>|)>>
    <frac|\<partial\>\<beta\><rsub|i>|\<partial\>z>
    \<delta\><rsub|i\<in\>M<rsub|\<beta\>>>,
  </equation*>

  For stability, using <math|\<sigma\><rprime|'>> instead of
  <math|\<sigma\>=1/<around*|\||w|\|>> where
  <math|\<sigma\>=ln<around*|(|1+exp<around*|(|\<sigma\><rprime|'>|)>|)>>???
  (C.f. <hlink|here|https://gist.github.com/altosaar/75df2f05698fb6872174306425bcc780>.)
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-10|<tuple|3.1|?>>
    <associate|auto-2|<tuple|1.1|?>>
    <associate|auto-3|<tuple|1.2|?>>
    <associate|auto-4|<tuple|2|?>>
    <associate|auto-5|<tuple|2.1|?>>
    <associate|auto-6|<tuple|2.2|?>>
    <associate|auto-7|<tuple|2.3|?>>
    <associate|auto-8|<tuple|2.4|?>>
    <associate|auto-9|<tuple|3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Preliminary>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Assumptions on Posterior
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|1.2<space|2spc>Bayesian Inference
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Neural
      Network for Posterior> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Model
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Numerical Consideration
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|2.3<space|2spc>Cost-Function (Performance)
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <with|par-left|<quote|1tab>|2.4<space|2spc>Gradient
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>>
    </associate>
  </collection>
</auxiliary>