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

  We we are going to do is fit this <math|p<around*|(|\<theta\>;D|)>> by ANN
  for any given <math|D>. To do so, we have to assume that
  <math|supp<around*|{|p<around*|(|\<theta\>;D|)>|}>=\<bbb-R\><rsup|d>> for
  some <math|d\<in\>\<bbb-N\><rsup|+>> (i.e. has no compact support) but
  decrease exponentially fast as <math|<around*|\<\|\|\>|\<theta\>|\<\|\|\>>\<rightarrow\>+\<infty\>>.
  With this assumption, we can use Gaussian function as the activation of the
  ANN. We propose the fitting function

  <\equation*>
    q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>=<big|sum><rsub|i=1><rsup|N<rsub|c>>w<rsub|i><around*|(|a|)><around*|{|<big|prod><rsub|j=1><rsup|d>\<Phi\><around*|(|\<theta\><rsub|j>-\<mu\><rsub|i
    j>,\<sigma\><around*|(|\<zeta\><rsub|i j>|)>|)>|}>,
  </equation*>

  where

  <\eqnarray*>
    <tformat|<table|<row|<cell|w<rsub|i><around*|(|a|)>>|<cell|=>|<cell|<frac|exp<around*|(|a<rsub|i>|)>|<big|sum><rsub|j=1><rsup|N>exp<around*|(|a<rsub|j>|)>>=softmax<around*|(|a<rsub|i>;a|)>;>>|<row|<cell|\<sigma\><around*|(|\<zeta\><rsub|i
    j>|)>>|<cell|=>|<cell|ln<around*|(|1+exp<around*|(|\<zeta\><rsub|i
    j>|)>|)>,>>>>
  </eqnarray*>

  and <math|a<rsub|i>,\<mu\><rsub|i j>,\<zeta\><rsub|i j>\<in\>\<bbb-R\>> for
  <math|\<forall\>i> and

  <\equation*>
    \<Phi\><around*|(|x-\<mu\>,\<sigma\>|)>\<assign\><sqrt|<frac|1|2 \<pi\>
    \<sigma\><rsup|2>>> exp<around*|(|-<frac|<around*|(|x-\<mu\>|)><rsup|2>|2
    \<sigma\><rsup|2>>|)>
  </equation*>

  being the Gaussian PDF. The introduction of <math|\<zeta\>> is for
  numerical consideration, see below.

  <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>> has probablitic
  illustration. <math|<big|prod><rsub|j=1><rsup|d>\<Phi\><around*|(|\<theta\><rsub|j>-\<mu\><rsub|i
  j>,\<sigma\><around*|(|\<zeta\><rsub|i j>|)>|)>> corresponds to
  multi-dimensional Gaussian distribution (denote
  <math|<with|math-font|cal|N>>), with all dimensions independent with each
  other. The <math|<around*|{|w<rsub|i><around*|(|a|)>|}>> is a categorical
  distribution, randomly choosing the Gaussian distributions. Thus
  <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>> is a composition:
  <math|categorical \<rightarrow\> Gaussian>. This is the <hlink|<em|mixture
  distribution>|https://en.wikipedia.org/wiki/Mixture_distribution>.

  Since there's no compact support, for both
  <math|p<around*|(|\<theta\>;D|)>> and <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>>,
  KL-divergence (equivalently, ELBO) can be safely employed as the
  cost-function of the fitting.

  <subsection|Numerical Consideration>

  For numerical consideration, instead of fitting
  <math|p<around*|(|\<theta\>;D|)>> by <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>>,
  we fit <math|ln p<around*|(|\<theta\>;D|)>> by <math|ln
  q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>>. To compute <math|ln
  q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>>, we have to employ some
  approximation method. Let

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<beta\><rsub|i>>|<cell|\<assign\>>|<cell|ln<around*|(|w<rsub|i><around*|(|a|)>
    <around*|{|<big|prod><rsub|j=1><rsup|d>\<Phi\><around*|(|\<theta\><rsub|j>-\<mu\><rsub|i
    j>,\<sigma\><around*|(|\<zeta\><rsub|i
    j>|)>|)>|}>|)>>>|<row|<cell|>|<cell|=>|<cell|ln
    w<rsub|i><around*|(|a|)>+<big|sum><rsub|j=1><rsup|d><around*|{|-<frac|1|2>
    ln<around*|(|2 \<pi\>|)>-ln<around*|(|\<sigma\><around*|(|\<zeta\><rsub|i
    j>|)>|)>-<frac|<around*|(|\<theta\><rsub|j>-\<mu\><rsub|i j>|)><rsup|2>|2
    \<sigma\><rsup|2><around*|(|\<zeta\><rsub|i j>|)>>|}>,>>>>
  </eqnarray*>

  thus <math|ln q=ln<around*|(|<big|sum><rsub|i=1><rsup|N<rsub|c>>exp<around*|(|\<beta\><rsub|i>|)>|)>.>
  We first compute all the <math|\<beta\><rsub|i>> and pick the maximum
  <math|\<beta\><rsub|max>>. Then,

  <\eqnarray*>
    <tformat|<table|<row|<cell|ln q>|<cell|=>|<cell|ln<around*|(|<big|sum><rsub|i=1><rsup|N<rsub|c>>exp<around*|(|\<beta\><rsub|i>-\<beta\><rsub|max>|)>|)>+\<beta\><rsub|max>>>|<row|<cell|>|<cell|=>|<cell|ln<around*|(|<big|sum><rsub|i=1><rsup|N<rsub|c>>exp<around*|(|<with|font-series|bold|\<delta\>\<beta\><rsub|i>>|)>|)>+\<beta\><rsub|max>,>>>>
  </eqnarray*>

  where <math|\<delta\>\<beta\><rsub|i>\<assign\>\<beta\><rsub|i>-\<beta\><rsub|max>>
  for convenience. This trick can also be applied to any softmax. Indeed,

  <\eqnarray*>
    <tformat|<table|<row|<cell|w<rsub|i><around*|(|a|)>>|<cell|=>|<cell|softmax<around*|(|a<rsub|i>;a|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|exp<around*|(|a<rsub|i>|)>|<big|sum><rsub|j=1><rsup|N<rsub|c>>exp<around*|(|a<rsub|j>|)>>,>>|<row|<cell|>|<cell|=>|<cell|<frac|exp<around*|(|\<delta\>a<rsub|i>|)>|<big|sum><rsub|j=1><rsup|N<rsub|c>>exp<around*|(|\<delta\>a<rsub|j>|)>>,>>>>
  </eqnarray*>

  where <math|\<delta\>a<rsub|i>\<assign\>a<rsub|i>-a<rsub|max>> for
  convenience. There are no numerical instability in expression now.

  <subsection|Cost-Function (Performance)>

  <\eqnarray*>
    <tformat|<table|<row|<cell|ELBO<around*|(|a,\<mu\>,\<zeta\>|)>>|<cell|\<assign\>>|<cell|\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>;w,b|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>-ln q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>|]>>>|<row|<cell|>|<cell|\<approx\>>|<cell|<around*|(|<frac|1|n>
    <big|sum><rsub|\<theta\><rsup|<around*|(|s|)>>>|)><around*|{|ln
    p<around*|(|\<theta\><rsub|<around*|(|s|)>>;D|)>-lnq<around*|(|\<theta\><rsub|<around*|(|s|)>>;a,\<mu\>,\<zeta\>|)>|}>,>>>>
  </eqnarray*>

  where <math|<around*|{|\<theta\><rsub|<around*|(|s|)>>: s=1,\<ldots\>,n|}>>
  is sampled from <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>> as a
  distribution.

  <subsection|Gradient>

  Let <math|z\<assign\><around*|(|a,\<mu\>,\<zeta\>|)>>. Then,

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>ELBO|\<partial\>z><around*|(|z|)>>|<cell|=>|<cell|<frac|\<partial\>|\<partial\>z><big|int>\<mathd\>\<theta\>
    q<around*|(|\<theta\>;z|)> <around*|{|ln p<around*|(|\<theta\>;D|)>-ln
    q<around*|(|\<theta\>;z|)>|}>>>|<row|<cell|>|<cell|=>|<cell|<big|int>\<mathd\>\<theta\>
    q<around*|(|\<theta\>;z|)> <frac|\<partial\>ln
    q|\<partial\>z><around*|(|\<theta\>;z|)> <around*|{|ln
    p<around*|(|\<theta\>;D|)>-ln q<around*|(|\<theta\>;z|)>-1|}>>>|<row|<cell|>|<cell|\<approx\>>|<cell|<around*|(|<frac|1|n>
    <big|sum><rsub|\<theta\><rsub|<around*|(|s|)>>>|)><frac|\<partial\>ln
    q|\<partial\>z><around*|(|\<theta\><rsub|<around*|(|s|)>>;z|)>
    <around*|{|ln p<around*|(|\<theta\><rsub|<around*|(|s|)>>;D|)>-ln
    q<around*|(|\<theta\><rsub|<around*|(|s|)>>;z|)>-1|}>>>>>
  </eqnarray*>

  where <math|<around*|{|\<theta\><rsub|<around*|(|s|)>>: s=1,\<ldots\>,n|}>>
  is sampled from <math|q<around*|(|\<theta\>;z|)>> as a distribution. Next,
  since <math|ln q=ln<around*|(|<big|sum><rsub|i=1><rsup|N<rsub|c>>exp<around*|(|\<beta\><rsub|i>|)>|)>>,
  we have

  <\equation*>
    <frac|\<partial\>ln q|\<partial\>z><around*|(|\<theta\>;z|)>=<big|sum><rsub|i=1><rsup|N<rsub|c>>softmax<around*|(|\<beta\><rsub|i>;\<beta\>|)>
    <frac|\<partial\>\<beta\><rsub|i>|\<partial\>z>.
  </equation*>

  To calculate <math|\<partial\>\<beta\><rsub|i>/\<partial\>a<rsub|k>>,
  <math|\<partial\>\<beta\><rsub|i>/\<partial\>\<mu\><rsub|j k>> and
  <with|font-series|bold|<math|\<partial\>\<beta\><rsub|i>/\<partial\>\<zeta\><rsub|j
  k>>>, recall

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<beta\><rsub|i>>|<cell|=>|<cell|ln
    w<rsub|i><around*|(|a|)>+<big|sum><rsub|j=1><rsup|d><around*|{|-<frac|1|2>
    ln<around*|(|2 \<pi\>|)>-ln<around*|(|\<sigma\><around*|(|\<zeta\><rsub|i
    j>|)>|)>-<frac|<around*|(|\<theta\><rsub|j>-\<mu\><rsub|i j>|)><rsup|2>|2
    \<sigma\><rsup|2><around*|(|\<zeta\><rsub|i
    j>|)>>|}>>>|<row|<cell|>|<cell|=>|<cell|ln<around*|(|<frac|exp<around*|(|a<rsub|i>|)>|<big|sum><rsub|j=1><rsup|N<rsub|c>>exp<around*|(|a<rsub|j>|)>>|)>+<big|sum><rsub|j=1><rsup|d><around*|{|-<frac|1|2>
    ln<around*|(|2 \<pi\>|)>-ln<around*|(|\<sigma\><around*|(|\<zeta\><rsub|i
    j>|)>|)>-<frac|<around*|(|\<theta\><rsub|j>-\<mu\><rsub|i j>|)><rsup|2>|2
    \<sigma\><rsup|2><around*|(|\<zeta\><rsub|i
    j>|)>>|}>>>|<row|<cell|>|<cell|=>|<cell|a<rsub|i>-ln<around*|(|<big|sum><rsub|j=1><rsup|N<rsub|c>>exp<around*|(|a<rsub|j>|)>|)>+<big|sum><rsub|j=1><rsup|d><around*|{|-<frac|1|2>
    ln<around*|(|2 \<pi\>|)>-ln<around*|(|\<sigma\><around*|(|\<zeta\><rsub|i
    j>|)>|)>-<frac|<around*|(|\<theta\><rsub|j>-\<mu\><rsub|i j>|)><rsup|2>|2
    \<sigma\><rsup|2><around*|(|\<zeta\><rsub|i j>|)>>|}>,>>>>
  </eqnarray*>

  we thus have

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>\<beta\><rsub|i>|\<partial\>a<rsub|i>>>|<cell|=>|<cell|1-<frac|exp<around*|(|a<rsub|i>|)>|<big|sum><rsub|j=1><rsup|N<rsub|c>>exp<around*|(|a<rsub|j>|)>>>>|<row|<cell|>|<cell|=>|<cell|1-softmax<around*|(|a<rsub|i>;a|)>>>|<row|<cell|<frac|\<partial\>\<beta\><rsub|i>|\<partial\>\<mu\><rsub|i
    j>>>|<cell|=>|<cell|-<frac|\<theta\><rsub|j>-\<mu\><rsub|i
    j>|\<sigma\><rsup|2><around*|(|\<zeta\><rsub|i
    j>|)>>;>>|<row|<cell|<frac|\<partial\>\<beta\><rsub|i>|\<partial\>\<zeta\><rsub|i
    j>>>|<cell|=>|<cell|<around*|{|1+<frac|<around*|(|\<theta\><rsub|j>-\<mu\><rsub|i
    j>|)><rsup|2>|\<sigma\><rsup|2><around*|(|\<zeta\><rsub|i
    j>|)>>|}>\<times\><frac|\<partial\>\<sigma\>|\<partial\>\<zeta\><rsub|i
    j>><around*|(|\<zeta\><rsub|i j>|)> <frac|1|\<sigma\><around*|(|\<zeta\><rsub|i
    j>|)>>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|\<sigma\><around*|(|\<zeta\><rsub|i
    j>|)>> <around*|{|1+<frac|<around*|(|\<theta\><rsub|j>-\<mu\><rsub|i
    j>|)><rsup|2>|\<sigma\><rsup|2><around*|(|\<zeta\><rsub|i j>|)>>|}>
    sigmoid<around*|(|\<zeta\><rsub|i j>|)>;>>>>
  </eqnarray*>

  and all others vanish.

  And recall

  <\equation*>
    <frac|\<partial\>ELBO|\<partial\>z><around*|(|z|)>\<approx\><around*|(|<frac|1|n>
    <big|sum><rsub|\<theta\><rsub|<around*|(|s|)>>>|)> <around*|{|ln
    p<around*|(|\<theta\><rsub|<around*|(|s|)>>;D|)>-ln
    q<around*|(|\<theta\><rsub|<around*|(|s|)>>;z|)>-1|}>
    <big|sum><rsub|i=1><rsup|N<rsub|c>>softmax<around*|(|\<beta\><rsub|i>;\<beta\>|)>
    <frac|\<partial\>\<beta\><rsub|i>|\<partial\>z>,
  </equation*>
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