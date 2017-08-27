<TeXmacs|1.99.1>

<style|generic>

<\body>
  <section|Preliminary>

  <subsection|Assumptions on Posterior>

  Let <math|f<around*|(|x;\<theta\>|)>> a function of <math|x> with parameter
  <math|\<theta\>>. Let <math|y=f<around*|(|x;\<theta\>|)>> an observable,
  thus the observed value obeys a Gaussian distribution. Thus, for a list of
  observations <math|D\<assign\><around*|{|<around*|(|x<rsub|i>,y<rsub|i>,\<sigma\><rsub|i>|)>:i=1,\<ldots\>,N<rsub|D>|}>>
  (<math|\<sigma\><rsub|i>> is the observational error of <math|y<rsub|i>>),
  we can construct a (logrithmic) likelihood, as

  <\eqnarray*>
    <tformat|<table|<row|<cell|ln p<around*|(|D\|\<theta\>|)>>|<cell|=>|<cell|ln<around*|(|<big|prod><rsub|i=1><rsup|N<rsub|D>><frac|1|<sqrt|2
    \<pi\> \<sigma\><rsub|i><rsup|2>>> exp<around*|{|-<frac|1|2>
    <around*|(|<frac|y<rsub|i>-f<around*|(|x<rsub|i>;\<theta\>|)>|\<sigma\><rsub|i>>|)><rsup|2>|}>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N<rsub|D>><around*|{|-<frac|1|2>ln
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

  <subsection|The Model>

  Suppose we have a model, <math|f<around*|(|x,\<theta\>|)>>, where <math|x>
  is the input and <math|\<theta\>> the set of parameters of this model. Let
  <math|D> denotes an arbitrarily given dataset, i.e.
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
  With this assumption, <math|ln p<around*|(|\<theta\>;D|)>> is well-defined.
  For ANN, we propose using Gaussian function as the activation-function.
  Thus, we have the fitting function

  <\equation*>
    q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>=<big|sum><rsub|i=1><rsup|N<rsub|c>>w<rsub|i><around*|(|a|)><around*|{|<big|prod><rsub|j=1><rsup|d>\<Phi\><around*|(|\<theta\><rsub|j>-\<mu\><rsub|i
    j>,\<sigma\><around*|(|\<zeta\><rsub|i j>|)>|)>|}>,
  </equation*>

  where

  <\eqnarray*>
    <tformat|<table|<row|<cell|w<rsub|i><around*|(|a|)>>|<cell|=>|<cell|<frac|exp<around*|(|a<rsub|i>|)>|<big|sum><rsub|j=1><rsup|N>exp<around*|(|a<rsub|j>|)>>=softmax<around*|(|i;a|)>;>>|<row|<cell|\<sigma\><around*|(|\<zeta\><rsub|i
    j>|)>>|<cell|=>|<cell|ln<around*|(|1+exp<around*|(|\<zeta\><rsub|i
    j>|)>|)>,>>>>
  </eqnarray*>

  and <math|a<rsub|i>,\<mu\><rsub|i j>,\<zeta\><rsub|i j>\<in\>\<bbb-R\>> for
  <math|\<forall\>i,\<forall\>j> and

  <\equation*>
    \<Phi\><around*|(|x-\<mu\>,\<sigma\>|)>\<assign\><sqrt|<frac|1|2 \<pi\>
    \<sigma\><rsup|2>>> exp<around*|(|-<frac|<around*|(|x-\<mu\>|)><rsup|2>|2
    \<sigma\><rsup|2>>|)>
  </equation*>

  being the Gaussian PDF. The introduction of <math|\<zeta\>> is for
  numerical consideration, see below.

  <subsubsection|Numerical Consideration>

  If, in <math|q>, we regard <math|w>, <math|\<mu\>>, and <math|\<sigma\>> as
  independent variables, then the only singularity appears at
  <math|\<sigma\>=0>. Indeed, <math|\<sigma\>> appears in <math|\<Phi\>> (as
  well as the derivatives of <math|\<Phi\>>) as denominator only, while
  others as numerators. However, once doing numerical iterations with a
  finite step-length of <math|\<sigma\>>, the probability of reaching or even
  crossing <math|0> point cannot be surely absent. This is how we may
  encounter this singularity in practice.

  Introducing the <math|\<zeta\>> is our trick of avoiding this singularity.
  Precisely, using a singular map that pushes the singularity to infinity
  solves the singularity. In this case, using <math|softplus<around*|(|.|)>>
  that pushes <math|\<sigma\>=0> to <math|\<zeta\>\<rightarrow\>-\<infty\>>,
  so that, with finite steps of iteration, singularity (at <math|-\<infty\>>)
  cannot be reached.

  This trick (i.e. pushing a singularity to infinity) is the same as in
  avoiding the horizon-singularity of Schwarzschild solution of black hole.

  <subsection|Interpretation>

  <subsubsection|As a Mixture Distribution>

  <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>> has a probablitic
  interpretation. <math|<big|prod><rsub|j=1><rsup|d>\<Phi\><around*|(|\<theta\><rsub|j>-\<mu\><rsub|i
  j>,\<sigma\><around*|(|\<zeta\><rsub|i j>|)>|)>> corresponds to
  multi-dimensional Gaussian distribution (denote
  <math|<with|math-font|cal|N>>), with all dimensions independent with each
  other. The <math|<around*|{|w<rsub|i><around*|(|a|)>|}>> is a categorical
  distribution, randomly choosing the Gaussian distributions. Thus
  <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>> is a composition:
  <math|categorical \<rightarrow\> Gaussian>. This is the <hlink|<em|mixture
  distribution>|https://en.wikipedia.org/wiki/Mixture_distribution>.

  <subsubsection|As a Generalization>

  This model can also be intrepreted as a direct generalization of
  <hlink|mean-field variational inference|https://arxiv.org/pdf/1601.00670.pdf>.
  Indeed, let <math|N<rsub|c>=1>, this model reduces to mean-field
  variational inference. Remark that mean-field variational inference is a
  mature algorithm and has been sucessfully established on many practical
  applications.

  <subsection|Cost-Function>

  <\eqnarray*>
    <tformat|<table|<row|<cell|ELBO<around*|(|a,\<mu\>,\<zeta\>|)>>|<cell|\<assign\>>|<cell|\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>;w,b|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>-ln q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>|]>>>|<row|<cell|>|<cell|\<approx\>>|<cell|<around*|(|<frac|1|n>
    <big|sum><rsub|\<theta\><rsup|<around*|(|s|)>>>|)><around*|{|ln
    p<around*|(|\<theta\><rsub|<around*|(|s|)>>;D|)>-lnq<around*|(|\<theta\><rsub|<around*|(|s|)>>;a,\<mu\>,\<zeta\>|)>|}>,>>>>
  </eqnarray*>

  where <math|<around*|{|\<theta\><rsub|<around*|(|s|)>>: s=1,\<ldots\>,n|}>>
  is sampled from <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>> as a
  distribution. Since there's no compact support for both
  <math|p<around*|(|\<theta\>;D|)>> and <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>>,
  <math|ELBO> is well-defined, as the cost-function (or say loss-function,
  performance, etc) of the fitting.
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-10|<tuple|2.3|?>>
    <associate|auto-2|<tuple|1.1|?>>
    <associate|auto-3|<tuple|1.2|?>>
    <associate|auto-4|<tuple|2|?>>
    <associate|auto-5|<tuple|2.1|?>>
    <associate|auto-6|<tuple|2.1.1|?>>
    <associate|auto-7|<tuple|2.2|?>>
    <associate|auto-8|<tuple|2.2.1|?>>
    <associate|auto-9|<tuple|2.2.2|?>>
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