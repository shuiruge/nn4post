<TeXmacs|1.99.1>

<style|generic>

<\body>
  <section|Notations>

  Let <math|f<around*|(|x;\<theta\>|)>> a function of <math|x\<in\>X> with
  parameter <math|\<theta\>\<in\>\<Theta\>>. The dimension of
  <math|\<Theta\>> is <math|N<rsub|d>>. Let
  <math|y=f<around*|(|x;\<theta\>|)>> an observable, thus the observed value
  obeys a Gaussian distribution. Let <math|D> denotes a set of observations,
  <math|D\<assign\><around*|{|<around*|(|x<rsub|i>,y<rsub|i>,\<sigma\><rsub|i>|)>:i=1,\<ldots\>,N<rsub|D>|}>>,
  wherein <math|x<rsub|i>> is the <math|i>th input, <math|y<rsub|i>> its
  observed value, and <math|\<sigma\><rsub|i>> the observational error of
  <math|y<rsub|i>>. We may employ mini-batch technique, thus denote
  <math|D<rsub|m>\<assign\><around*|{|<around*|(|x<rsub|i>,y<rsub|i>,\<sigma\><rsub|i>|)>:i=1,\<ldots\>,N<rsub|m>|}>\<subset\>D>
  as a mini-batch, with batch-size <math|N<rsub|m>\<leqslant\>N<rsub|D>>. We
  use <math|\<bbb-E\><rsub|f<around*|(|\<theta\>|)>><around*|[|g<around*|(|\<theta\>|)>|]>>
  represent the expectation of function <math|g> of a random variable obeys
  the p.d.f. <math|f>. <math|\<Phi\>> is for Gaussian p.d.f.

  Later we will introduce variables <math|a<rsub|i>>, <math|\<mu\><rsub|i
  \<alpha\>>>, and <math|\<zeta\><rsub|i \<alpha\>>>, where
  <math|i=1,\<ldots\>,N<rsub|c>> (defined later) and
  <math|\<alpha\>=1,\<ldots\>,N<rsub|d>>. Let
  <math|z\<assign\><around*|(|a,\<mu\>,\<zeta\>|)>>; and for
  <math|\<forall\>i> given, <math|z<rsub|i>\<assign\><around*|(|a<rsub|i>,\<mu\><rsub|i\<alpha\>>,\<zeta\><rsub|i\<alpha\>>|)>>
  for <math|\<forall\>\<alpha\>>. Define space
  <math|Z\<assign\><around*|{|\<forall\>z|}>>; and for <math|\<forall\>i>
  given, define its subspace <math|Z<rsub|i>\<assign\><around*|{|\<forall\>z<rsub|i>|}>>.

  <section|Neural Network for Posterior (nn4post)>

  <subsection|The Model>

  Suppose we have some prior on <math|\<theta\>>,
  <math|p<around*|(|\<theta\>|)>>, we gain the un-normalized posterior
  <math|p<around*|(|D\|\<theta\>|)> p<around*|(|\<theta\>|)>>. With <math|D>
  arbitrarily given, this un-normalized posterior is a function of
  <math|\<theta\>>, denoted by <math|p<around*|(|\<theta\>;D|)>><\footnote>
    This is why we use `` <math|;> '' instead of `` <math|,> '', indicating
    that <math|D> has been (arbitrarily) given and fixed.
  </footnote>.<\footnote>
    The normalized posterior <math|p<around*|(|\<theta\>\|D|)>=p<around*|(|D\|\<theta\>|)>
    p<around*|(|\<theta\>|)>/p<around*|(|D|)>=p<around*|(|\<theta\>;D|)>/p<around*|(|D|)>>,
    by Bayes's rule.
  </footnote>

  We we are going to do is fit this <math|p<around*|(|\<theta\>;D|)>> by ANN
  for any given <math|D>. To do so, we have to assume that
  <math|supp<around*|{|p<around*|(|\<theta\>;D|)>|}>=\<bbb-R\><rsup|d>> for
  some <math|d\<in\>\<bbb-N\><rsup|+>> (i.e. has no compact support) but
  decrease exponentially fast as <math|<around*|\<\|\|\>|\<theta\>|\<\|\|\>>\<rightarrow\>+\<infty\>>.
  With this assumption, <math|ln p<around*|(|\<theta\>;D|)>> is well-defined.
  For ANN, we propose using Gaussian function as the activation-function.
  Thus, we have the fitting function

  <\equation>
    q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>\<assign\><big|sum><rsub|i=1><rsup|N<rsub|c>>c<rsub|i><around*|(|a|)><around*|{|<big|prod><rsub|\<alpha\>=1><rsup|d>\<Phi\><around*|(|\<theta\><rsub|\<alpha\>>-\<mu\><rsub|i
    \<alpha\>>,\<sigma\><around*|(|\<zeta\><rsub|i \<alpha\>>|)>|)>|}>,
  </equation>

  where

  <\eqnarray*>
    <tformat|<table|<row|<cell|c<rsub|i><around*|(|a|)>>|<cell|=>|<cell|<frac|exp<around*|(|a<rsub|i>|)>|<big|sum><rsub|j=1><rsup|N<rsub|c>>exp<around*|(|a<rsub|j>|)>>=softmax<around*|(|i;a|)>;<eq-number>>>|<row|<cell|\<sigma\><around*|(|\<zeta\><rsub|i
    \<alpha\>>|)>>|<cell|=>|<cell|ln<around*|(|1+exp<around*|(|\<zeta\><rsub|i
    \<alpha\>>|)>|)>,<eq-number>>>>>
  </eqnarray*>

  and <math|a<rsub|i>,\<mu\><rsub|i \<alpha\>>,\<zeta\><rsub|i
  \<alpha\>>\<in\>\<bbb-R\>> for <math|\<forall\>i,\<forall\>\<alpha\>>, and

  <\equation>
    \<Phi\><around*|(|x;\<mu\>,\<sigma\>|)>\<assign\><sqrt|<frac|1|2 \<pi\>
    \<sigma\><rsup|2>>> exp<around*|(|-<frac|<around*|(|x-\<mu\>|)><rsup|2>|2
    \<sigma\><rsup|2>>|)>
  </equation>

  being the Gaussian p.d.f.. The introduction of <math|\<zeta\>> is for
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

  <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>> has a probabilistic
  interpretation. <math|<big|prod><rsub|j=1><rsup|d>\<Phi\><around*|(|\<theta\><rsub|j>-\<mu\><rsub|i
  j>,\<sigma\><around*|(|\<zeta\><rsub|i j>|)>|)>> corresponds to
  multi-dimensional Gaussian distribution (denote
  <math|<with|math-font|cal|N>>), with all dimensions independent with each
  other. The <math|<around*|{|c<rsub|i><around*|(|a|)>|}>> is a categorical
  distribution, randomly choosing the Gaussian distributions. Thus
  <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>> is a composition:
  <math|categorical \<rightarrow\> Gaussian>. This is the <hlink|<em|mixture
  distribution>|https://en.wikipedia.org/wiki/Mixture_distribution>.

  <subsubsection|As a Generalization>

  This model can also be interpreted as a direct generalization of
  <hlink|mean-field variational inference|https://arxiv.org/pdf/1601.00670.pdf>.
  Indeed, let <math|N<rsub|c>=1>, this model reduces to mean-field
  variational inference. Remark that mean-field variational inference is a
  mature algorithm and has been successfully established on many practical
  applications.

  <subsubsection|As a Neural Network>

  <subsection|Marginalization>

  This model can be marginalized easily. This then benefits the transfering
  of the model components. Precisely, for any dimension-index <math|\<beta\>>
  given, we can marginalize all other dimensions directly, leaving

  <\eqnarray*>
    <tformat|<table|<row|<cell|q<around*|(|\<theta\><rsub|\<beta\>>;a,\<mu\>,\<zeta\>|)>>|<cell|=>|<cell|<big|prod><rsub|\<forall\>\<gamma\>\<neq\>\<beta\>><big|int>\<mathd\>\<theta\><rsub|\<gamma\>>
    <big|sum><rsub|i=1><rsup|N<rsub|c>>c<rsub|i><around*|(|a|)><around*|{|<big|prod><rsub|\<alpha\>=1><rsup|d>\<Phi\><around*|(|\<theta\><rsub|\<alpha\>>;\<mu\><rsub|i
    \<alpha\>>,\<sigma\><around*|(|\<zeta\><rsub|i
    \<alpha\>>|)>|)>|}><eq-number>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N<rsub|c>>c<rsub|i><around*|(|a|)>
    \<Phi\><around*|(|\<theta\><rsub|\<beta\>>;\<mu\><rsub|i
    \<beta\>>,\<sigma\><around*|(|\<zeta\><rsub|i
    \<beta\>>|)>|)>,<eq-number>>>>>
  </eqnarray*>

  where employed the normalization of <math|\<Phi\>>.

  <subsection|Loss-Function>

  We employ <hlink|\Pevidence of lower bound\Q
  (ELBO)|http://www.umiacs.umd.edu/~xyang35/files/understanding-variational-lower.pdf><\footnote>
    The relation between ELBO and KL-divergence is that
    <math|ELBO=-KL<around*|(|q\<\|\|\>p|)>+Const>.
  </footnote>. It is ensured to have a unique global maximum, at which
  <math|p<around*|(|\<theta\>;D|)>=q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>>.

  <\equation*>
    ELBO<around*|(|a,\<mu\>,\<zeta\>|)>\<assign\>\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>-ln q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>|]>.
  </equation*>

  Since there's no compact support for both <math|p<around*|(|\<theta\>;D|)>>
  and <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>>, <math|ELBO> is
  well-defined. The loss-function (or say loss-function, performance, etc) of
  the fitting is then defined as <math|<with|math-font|cal|L>\<assign\>-ELBO>,
  i.e.

  <\equation*>
    <with|math-font|cal|<with|math-font|cal|L><around*|(|a,\<mu\>,\<zeta\>|)>>=-\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>-ln q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>|]>,
  </equation*>

  or, recall <math|\<bbb-H\><around*|[|q|]>\<assign\>-\<bbb-E\><rsub|q><around*|[|ln
  q|]>> for any distribution <math|q>,

  <\equation*>
    <with|math-font|cal|<with|math-font|cal|L><around*|(|a,\<mu\>,\<zeta\>|)>>=-\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>|]>-\<bbb-H\><around*|[|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>|]>.
  </equation*>

  <subsection|Relation with the Traditional MAP Loss>

  <math|\<theta\><rsub|\<ast\>>=argmin<rsub|\<theta\>> <around*|{|-ln
  p<around*|(|\<theta\>;D|)>|}>>.

  Set <math|N<rsub|c>=1> and for <math|\<forall\>\<alpha\>>
  <math|\<zeta\><rsub|\<alpha\>>\<rightarrow\>-\<infty\>> so that
  <math|\<sigma\><rsub|\<alpha\>><around*|(|\<zeta\>|)>\<rightarrow\>0>, we
  get <math|q<around*|(|\<theta\>;z|)>\<rightarrow\>\<delta\><around*|(|\<theta\>-z|)>>.

  <\eqnarray*>
    <tformat|<table|<row|<cell|<with|math-font|cal|<with|math-font|cal|L><around*|(|z|)>>>|<cell|=>|<cell|-\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>;z|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>-ln q<around*|(|\<theta\>;z|)>|]>>>|<row|<cell|>|<cell|=>|<cell|-<big|int>\<mathd\>\<theta\>
    q<around*|(|\<theta\>;z|)> <around*|[|ln p<around*|(|\<theta\>;D|)>-ln
    q<around*|(|\<theta\>;z|)>|]>>>|<row|<cell|>|<cell|=>|<cell|-<big|int>\<mathd\>\<theta\>
    \<delta\><around*|(|\<theta\>-z|)> <around*|[|ln
    p<around*|(|\<theta\>;D|)>-ln \<delta\><around*|(|\<theta\>-z|)>|]>>>|<row|<cell|>|<cell|=>|<cell|-ln
    p<around*|(|z;D|)>+Const>>|<row|<cell|>|<cell|=>|<cell|<with|math-font|cal|L><rsub|MAP><around*|(|z|)>+Const>>>>
  </eqnarray*>

  <subsection|Relation between Relative Error of Inference and Loss>

  For the arbitrary model <math|y=f<around*|(|x;\<theta\>|)>>, for
  <math|\<forall\>x>, Bayesian inference gives prediction,
  <math|<around*|\<langle\>|f|\<rangle\>><around*|(|x|)>>, as

  <\equation*>
    <around*|\<langle\>|f|\<rangle\>><around*|(|x|)>\<assign\>\<bbb-E\><rsub|\<theta\>\<sim\>p<around*|(|\<theta\>\|D|)>><around*|[|f<around*|(|x;\<theta\>|)>|]>.
  </equation*>

  Since <math|q<around*|(|.;z|)>> (<math|z> as the parameter of <math|q>) is
  an approximation to <math|p<around*|(|.\|D|)>>, let
  <math|<around*|\<langle\>|f|\<rangle\>><rsub|q><around*|(|x|)>\<assign\>\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>;z|)>><around*|[|f<around*|(|x;\<theta\>|)>|]>>,
  then the difference between them is <math|\<delta\><around*|\<langle\>|f|\<rangle\>><around*|(|x|)>\<assign\><around*|\<langle\>|f|\<rangle\>><around*|(|x|)>-<around*|\<langle\>|f|\<rangle\>><rsub|q><around*|(|x|)>>.

  <\theorem>
    <label|theorem: Relation between Relative Error of Inference and Loss>We
    have the relation of order between the relative error of inference and
    loss

    <\equation*>
      <frac|\<delta\><around*|\<langle\>|f|\<rangle\>>|<around*|\<langle\>|f|\<rangle\>>><around*|(|x|)>\<sim\><with|math-font|cal|L>.
    </equation*>
  </theorem>

  <\proof>
    By definition, <math|\<delta\><around*|\<langle\>|f|\<rangle\>><around*|(|x|)>=<big|int>\<mathd\>\<theta\>
    f<around*|(|x;\<theta\>|)> <around*|[|p<around*|(|\<theta\>\|D|)>-q<around*|(|\<theta\>|)>|]>>.
    Thus

    <\eqnarray*>
      <tformat|<table|<row|<cell|\<delta\><around*|\<langle\>|f|\<rangle\>><around*|(|x|)>>|<cell|=>|<cell|<big|int>\<mathd\>\<theta\>
      q<around*|(|\<theta\>|)> f<around*|(|x;\<theta\>|)>
      <frac|p<around*|(|\<theta\>\|D|)>-q<around*|(|\<theta\>|)>|q<around*|(|\<theta\>|)>>>>|<row|<cell|>|<cell|=>|<cell|<big|int>\<mathd\>\<theta\>
      q<around*|(|\<theta\>|)> <around*|(|<frac|p<around*|(|\<theta\>\|D|)>|q<around*|(|\<theta\>|)>>-1|)>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>|)>><around*|[|f<around*|(|x;\<theta\>|)>
      <around*|(|<frac|p<around*|(|\<theta\>\|D|)>|q<around*|(|\<theta\>|)>>-1|)>|]>.>>>>
    </eqnarray*>

    Then, we have the relation of order

    <\equation*>
      <frac|\<delta\><around*|\<langle\>|f|\<rangle\>>|<around*|\<langle\>|f|\<rangle\>>><around*|(|x|)>\<sim\>\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>|)>><around*|[|<frac|p<around*|(|\<theta\>\|D|)>|q<around*|(|\<theta\>|)>>-1|]>
    </equation*>

    On the other hand, if <math|p<around*|(|.\|D|)>\<approx\>q<around*|(|.|)>>
    as we expect for <math|q<around*|(|.|)>>, then we have

    <\equation*>
      ln<around*|(|<frac|p<around*|(|\<theta\>\|D|)>|q<around*|(|\<theta\>|)>>|)>=ln<around*|(|<around*|[|<frac|p<around*|(|\<theta\>\|D|)>|q<around*|(|\<theta\>|)>>-1|]>+1|)>\<approx\><frac|p<around*|(|\<theta\>\|D|)>|q<around*|(|\<theta\>|)>>-1.
    </equation*>

    Thus,

    <\equation*>
      <frac|\<delta\><around*|\<langle\>|f|\<rangle\>>|<around*|\<langle\>|f|\<rangle\>>><around*|(|x|)>\<sim\>\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>|)>><around*|[|<frac|p<around*|(|\<theta\>\|D|)>|q<around*|(|\<theta\>|)>>-1|]>\<approx\>\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>|)>><around*|[|ln<around*|(|<frac|p<around*|(|\<theta\>\|D|)>|q<around*|(|\<theta\>|)>>|)>|]>=<with|math-font|cal|L>.
    </equation*>
  </proof>

  <section|Optimization>

  <subsection|ADVI>

  Automatic differentation variational inference (ADVI)<\footnote>
    See, <hlink|Kucukelbir, et al, 2016|https://arxiv.org/abs/1603.00788>.
  </footnote> has the advantage that the variance of its Monte Carlo integral
  is orderly smaller than that of black box variational inference (i.e.
  optimization directly using ELBO without further reparameterization).

  <subsubsection|Derivation>

  Precisely, recall <math|\<bbb-E\>> for mean value, <math|\<Phi\>> for
  Gaussian p.d.f., <math|\<sigma\><around*|(|.|)>> for softplus function,
  <math|c<around*|(|.|)>> for softmax function, and

  <\equation>
    q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>=<big|sum><rsub|i=1><rsup|N<rsub|c>>c<rsub|i><around*|(|a|)>
    \<Phi\><around*|(|\<theta\>;\<mu\><rsub|i>,\<sigma\><around*|(|\<zeta\><rsub|i>|)>|)>,
  </equation>

  we have, for any function <math|f>,

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<bbb-E\><rsub|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>><around*|[|f<around*|(|\<theta\>|)>|]>>|<cell|=>|<cell|<big|int>\<mathd\>\<theta\>
    <big|sum><rsub|i=1><rsup|N<rsub|c>>c<rsub|i><around*|(|a|)>
    \<Phi\><around*|(|\<theta\>;\<mu\><rsub|i>,\<sigma\><around*|(|\<zeta\><rsub|i>|)>|)>
    f<around*|(|\<theta\>|)><eq-number>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N<rsub|c>>c<rsub|i><around*|(|a|)>
    <big|int>\<mathd\>\<theta\> \<Phi\><around*|(|\<theta\>;\<mu\><rsub|i>,\<sigma\><around*|(|\<zeta\><rsub|i>|)>|)>
    f<around*|(|\<theta\>|)><eq-number>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N<rsub|c>>c<rsub|i><around*|(|a|)>
    \<bbb-E\><rsub|\<Phi\><around*|(|\<theta\>;\<mu\><rsub|i>,\<sigma\><around*|(|\<zeta\><rsub|i>|)>|)>><around*|[|f<around*|(|\<theta\>|)>|]>.<eq-number>>>>>
  </eqnarray*>

  With this general relation, we get

  <\eqnarray*>
    <tformat|<table|<row|<cell|<with|math-font|cal|<with|math-font|cal|L><around*|(|a,\<mu\>,\<zeta\>|)>>>|<cell|=>|<cell|-<around*|{|\<bbb-E\><rsub|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>|]>-\<bbb-E\><rsub|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>><around*|[|ln
    q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>|]>|}><eq-number>>>|<row|<cell|>|<cell|=>|<cell|-<big|sum><rsub|i><rsup|N<rsub|c>>c<rsub|i><around*|(|a|)>
    \<bbb-E\><rsub|\<Phi\><rsub|i><around*|(|\<theta\>;\<mu\><rsub|i>,\<sigma\><around*|(|\<zeta\><rsub|i>|)>|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>-ln q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>|]><eq-number>>>>>
  </eqnarray*>

  Then, for <math|\<forall\>i=1,\<ldots\>,N<rsub|c>>,
  <math|\<forall\>\<alpha\>=1,\<ldots\>,N<rsub|d>>, let

  <\equation>
    \<eta\><rsub|\<alpha\>>\<assign\><frac|\<theta\><rsub|\<alpha\>>-\<mu\><rsub|i\<alpha\>>|\<sigma\><around*|(|\<zeta\><rsub|i\<alpha\>>|)>>,
  </equation>

  we have

  <\equation>
    \<theta\><rsub|\<alpha\>>=\<sigma\><around*|(|\<zeta\><rsub|i\<alpha\>>|)>
    \<eta\><rsub|\<alpha\>>+\<mu\><rsub|i\<alpha\>>
  </equation>

  (or <math|\<theta\>=\<sigma\><around*|(|\<zeta\><rsub|i>|)>
  \<eta\>+\<mu\><rsub|i>> if hide the <math|\<alpha\>> index). So, for any
  <math|i>-component, we transform

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<theta\>>|<cell|\<rightarrow\>>|<cell|\<sigma\><around*|(|\<zeta\><rsub|i>|)>
    \<eta\>+\<mu\><rsub|i>;<eq-number>>>|<row|<cell|\<bbb-E\><rsub|\<Phi\><around*|(|\<theta\>;\<mu\><rsub|i>,\<sigma\><around*|(|\<zeta\><rsub|i>|)>|)>><around*|[|f<around*|(|\<theta\>|)>|]>>|<cell|\<rightarrow\>>|<cell|\<bbb-E\><rsub|\<Phi\><around*|(|\<eta\>;0,1|)>><around*|[|f<around*|(|\<sigma\><around*|(|\<zeta\><rsub|i>|)>
    \<eta\>+\<mu\><rsub|i>|)>|]>,<eq-number>>>>>
  </eqnarray*>

  where function <math|f> is arbitrary, thus holds for both <math|ln
  p<around*|(|.;D|)>> and <math|ln q<around*|(|.;a,\<mu\>,\<zeta\>|)>>.

  With this setting, the derivatives to <math|\<mu\>> and to <math|\<zeta\>>
  are completely independent of <math|\<bbb-E\><around*|[|.|]>>. And now, the
  loss function becomes

  <\equation*>
    <with|math-font|cal|L><around*|(|a,\<mu\>,\<zeta\>|)>=-<big|sum><rsub|i><rsup|N<rsub|c>>c<rsub|i><around*|(|a|)>
    \<bbb-E\><rsub|\<Phi\><around*|(|\<eta\>;0,1|)>><around*|[|ln
    p<around*|(|\<sigma\><around*|(|\<zeta\><rsub|i>|)>
    \<eta\>+\<mu\><rsub|i>;D|)>-ln q<around*|(|\<sigma\><around*|(|\<zeta\><rsub|i>|)>
    \<eta\>+\<mu\><rsub|i>;a,\<mu\>,\<zeta\>|)>|]>.
  </equation*>

  <subsection|Redefination of Gradients>

  <subsubsection|Gauge Fixing>

  Let <math|\<Delta\>t> the learning-rate. Then the updation of
  <math|a<rsub|i>> at one iteration by gradient decent method is

  <\equation*>
    \<Delta\>a<rsub|i>=-<frac|\<partial\><with|math-font|cal|L>|\<partial\>a<rsub|i>><around*|(|a,\<mu\>,\<zeta\>|)>
    \<Delta\>t.
  </equation*>

  Notice that redefining the <math|\<partial\><with|math-font|cal|L>/\<partial\>a>
  by

  <\equation*>
    <frac|\<partial\><with|math-font|cal|L>|\<partial\>a<rsub|i>><around*|(|a,\<mu\>,\<zeta\>|)>\<rightarrow\><frac|\<partial\><with|math-font|cal|L>|\<partial\>a<rsub|i>><around*|(|a,\<mu\>,\<zeta\>|)>+C,
  </equation*>

  where <math|C> can be any constant, leaves the updation of
  <math|c<rsub|i><around*|(|a|)>> invariant, since it makes

  <\equation*>
    \<Delta\>a<rsub|i>\<rightarrow\>-<frac|\<partial\><with|math-font|cal|L>|\<partial\>a<rsub|i>><around*|(|a,\<mu\>,\<zeta\>|)>
    \<Delta\>t-C \<Delta\>t,
  </equation*>

  thus

  <\equation*>
    c<rsub|i><around*|(|a+\<Delta\>a|)>\<rightarrow\><frac|exp<around*|(|a<rsub|i>+\<Delta\>a<rsub|i>-C
    \<Delta\>t|)>|<big|sum><rsub|j>exp<around*|(|a<rsub|j>+\<Delta\>a<rsub|j>-C
    \<Delta\>t|)>>=<frac|exp<around*|(|a<rsub|i>+\<Delta\>a<rsub|i>|)>|<big|sum><rsub|j>exp<around*|(|a<rsub|j>+\<Delta\>a<rsub|j>|)>>=c<rsub|i><around*|(|a+\<Delta\>a|)>.
  </equation*>

  This <math|C> thus provides an additional dof.<\footnote>
    As CL explained, the <math|c>s have less dofs as they look, since
    <math|<big|sum><rsub|i><rsup|N<rsub|c>>c<rsub|i>=1>. This restriction can
    provides an additional gauge. And the new dof <math|C> fixes this gauge.
  </footnote> We can tune the value of <math|C> so that the updation of
  <math|a<rsub|i>> is numerically stable. Indeed, let <math|C> be the average
  of <math|<around*|{|\<partial\><with|math-font|cal|L>/\<partial\>a<rsub|i>:i=1,\<ldots\>,N<rsub|c>|}>>,
  we find a pretty stability of <math|a> as well as a pretty accuracy of
  <math|c> in the iteration process of optimization, as the experiment on
  Gaussian mixture model shows.

  This motives us to, instead of modifying gradients, re-write
  <math|<with|math-font|cal|L>> by replacing the <math|c> in it by

  <\equation*>
    c<rsub|i><around*|(|a|)>\<rightarrow\>c<rsub|i><around*|(|a-<frac|<big|sum><rsub|j><rsup|N<rsub|c>>a<rsub|j>|N<rsub|c>>|)>.
  </equation*>

  Thus

  <\equation*>
    <frac|\<partial\>c<rsub|i>|\<partial\>a<rsub|k>><around*|(|a|)>\<rightarrow\>\<partial\><rsub|k>c<rsub|i><around*|(|a-<frac|<big|sum><rsub|j><rsup|N<rsub|c>>a<rsub|j>|N<rsub|c>>|)>
    -<frac|1|N<rsub|c>><big|sum><rsub|k><rsup|N<rsub|c>>\<partial\><rsub|k>c<rsub|i><around*|(|a-<frac|<big|sum><rsub|j><rsup|N<rsub|c>>a<rsub|j>|N<rsub|c>>|)>.
  </equation*>

  These two approaches are almost the same. But when <math|N<rsub|d>> is
  great enough, the difference between them raises. Indeed, experiments on
  Gaussian mixture distribution (as target) shows that the later converges
  apperately faster than the first.<\footnote>
    Why so?
  </footnote> Additionally, the second approach provides stability for
  softmax function, since the input of softmax is regular no matter how great
  the <math|a> is. So, we will use the later approach, i.e. modify the
  relation <math|c<rsub|i><around*|(|a|)>> in loss directly.

  <subsubsection|Re-scaling of <math|a>>

  In the optimization process, the scales of searching region of <math|a> and
  of <math|\<mu\>> and <math|\<zeta\>> may be different in order. So, there
  shall be an additional hyper-parameter for the re-scaling of <math|a>. The
  re-scaling factor, constant <math|r>, redefines

  <\equation*>
    c<rsub|i><around*|(|a|)>\<assign\>softmax<around*|(|i,r a|)>.
  </equation*>

  Tuning this additional hyper-parameter can ``normalize'' <math|a> to the
  same order of scale as <math|\<mu\>> and <math|\<zeta\>>, thus may improve
  the optimization.

  This rescaling, if dynamically (e.g. set <math|r> as <cpp|tf.placeholder>
  in TensorFlow), also helps fasten the speed of convergence. Indeed,
  especially with a large <math|N<rsub|d>>, the searching of targets
  <math|\<mu\>> and <math|\<zeta\>> lasts longer, so that <math|a>, as nearly
  random moving at this epoch, can be greatly dispersed, i.e.
  <math|max<around*|(|a|)>\<gg\>min<around*|(|a|)>>. As a result, when the
  targets <math|\<mu\>> and <math|\<zeta\>> have been reached, it needs
  extremely large number of iterations for <math|a> so that the target value
  (generally not so dispersed) can be reached. However, if <math|r> is
  inserted and tuned dynamically, setting <math|r\<rightarrow\>0> at the
  early searching (of targets <math|\<mu\>> and <math|\<zeta\>>) epoch, and
  then setting <math|r\<rightarrow\>1> after variables <math|\<mu\>> and
  <math|\<zeta\>> becoming slowly varying, meaning that the their targets
  have been reached. This thus largely speed up the convergence.

  <subsubsection|Frozen-out Problem>

  Generally we hope that the gradients diminish when and only when the
  optimization converges. However, even far from convergence, a tiny
  <math|c<rsub|i>> will diminish all the derivatives in the
  <math|i>-component, e.g. derivatives of <math|a<rsub|i>>,
  <math|\<mu\><rsub|i\<alpha\>>>, <math|\<zeta\><rsub|i\<alpha\>>>, since all
  these derivatives are proportional to <math|c<rsub|i>>.

  This problem can be solved by replacing, in the gradients, that

  <\equation*>
    <frac|\<partial\><with|math-font|cal|L>|\<partial\>z<rsub|i>>\<rightarrow\><frac|\<partial\><with|math-font|cal|L>|\<partial\>z<rsub|i>>
    <frac|1|c<rsub|i><around*|(|a|)>+\<epsilon\>>,
  </equation*>

  where <math|\<epsilon\>> is a tiny number for numerial stability as
  usual<\footnote>
    You may wonder why not set <math|\<epsilon\>=0>, since
    <math|c<around*|(|a|)>> is always non-vanishing. This concerns with the
    numerical instability in practice. Indeed, in TensorFlow, letting
    <math|\<epsilon\>=0> causes <cpp|NaN> after about <math|1000> iterations.
  </footnote>. This is valid since <math|c<rsub|i><around*|(|a|)>> are all
  positive. This modifies the direction of gradients in the space <math|Z>,
  but holds the same diection in each <math|i>-subspace <math|Z<rsub|i>>
  individually. And if <math|<around*|(|\<partial\><with|math-font|cal|L>/\<partial\>z<rsub|i>|)>/<around*|(|c<rsub|i><around*|(|a|)>+\<epsilon\>|)>=0>,
  we will have <math|\<partial\><with|math-font|cal|L>/\<partial\>z<rsub|i>=0>,
  meaning that both gradients leads to the same converge-point on the space
  <math|Z>. So, this modification speeds up the convergence without changing
  the converge-point.

  Generally, we can set

  <\equation*>
    <frac|\<partial\><with|math-font|cal|L>|\<partial\>z<rsub|i>>\<rightarrow\><frac|\<partial\><with|math-font|cal|L>|\<partial\>z<rsub|i>>
    <around*|(|<frac|1|c<rsub|i><around*|(|a|)>+\<epsilon\>>|)><rsup|\<beta\>>,
  </equation*>

  where <math|\<beta\>\<in\><around*|[|0,1|]>>. If <math|\<beta\>=1>, then
  back to the previous case; and if <math|\<beta\>=0>, then
  <math|\<partial\><with|math-font|cal|L>/\<partial\>z<rsub|i>> transforms
  nothing. Running <math|\<beta\>> in range <math|<around*|[|0,1|]>> then
  smoothly waving the transformation of <math|\<partial\><with|math-font|cal|L>/\<partial\>z<rsub|i>>.

  In TensorFlow, <math|\<epsilon\>> is usually set as
  <verbatim|1e-08><\footnote>
    C.f. <hlink|https://www.tensorflow.org/api_docs/python/tf/keras/backend/epsilon|https://www.tensorflow.org/api_docs/python/tf/keras/backend/epsilon>.
  </footnote>. However, the <math|c<around*|(|a|)>> can reach the order
  <verbatim|1e-24> in practice. (The reason why <math|\<epsilon\>> cannot be
  vanished is in footnote.) So the frozen-out problem can still remain, since
  even though transform as <math|\<partial\><with|math-font|cal|L>/\<partial\>z\<propto\>c<around*|(|a|)>\<sim\>10<rsup|-24>\<rightarrow\>\<partial\><with|math-font|cal|L>/\<partial\>z\<propto\>c<around*|(|a|)>/<around*|(|c<around*|(|a|)>+\<epsilon\>|)>\<approx\>c<around*|(|a|)>/\<epsilon\>\<sim\>10<rsup|-16>>,
  <math|\<partial\><with|math-font|cal|L>/\<partial\>z> is extremely tiny
  still. This can be solved by additionally clipping <math|c<around*|(|a|)>>
  by <math|\<epsilon\>> as the minimal value. Explicitly, after

  <\verbatim-code>
    a_mean = tf.reduce_mean(a, name='a_mean') \ # for gauge fixing.

    c = tf.softmax(r * (a - a_mean), name='c') \ # rescaling of `a`.
  </verbatim-code>

  additionally set (notice <math|c<around*|(|a|)>\<less\>1> always)

  <\verbatim-code>
    c = tf.clip_by_value(c, _EPSILON, 1, name='c_clipped')
  </verbatim-code>

  Or instead directly clipping on <math|a>? Indeed we can, but by clipping
  the gradient of <math|a>, instead of <math|a> itself. What we hope is that

  <\equation*>
    c<rsub|i><around*|(|a|)>\<equiv\><frac|exp<around*|(|a<rsub|i>-<big|sum><rsub|k><rsup|N<rsub|c>>a<rsub|k>/N<rsub|c>|)>|<big|sum><rsub|j><rsup|N<rsub|c>>exp<around*|(|a<rsub|j>-<big|sum><rsub|k><rsup|N<rsub|c>>a<rsub|k>/N<rsub|c>|)>>=<frac|exp<around*|(|a<rsub|i>|)>|<big|sum><rsub|j><rsup|N<rsub|c>>exp<around*|(|a<rsub|j>|)>>\<geqslant\>\<epsilon\>
  </equation*>

  for some <math|\<epsilon\>> as the ``accuracy of <math|c>'' (thus named as
  <verbatim|_C_ACCURACY> in code), which may different from the previous
  <math|\<epsilon\>> (i.e. the <verbatim|_EPSILON>) for numerical stability
  in dividing, but shall have <verbatim|_C_ACCURACY \<gtr\> _EPSILON>.

  gives

  <\equation*>
    a<rsub|i>\<geqslant\>ln<around*|(|\<epsilon\>|)>+ln<around*|(|<big|sum><rsub|j><rsup|N<rsub|c>>exp<around*|(|a<rsub|j>|)>|)>.
  </equation*>

  To ensure this, for some <math|\<epsilon\>> and some <math|a> given, define

  <\equation*>
    a<rsub|min>\<assign\>ln<around*|(|\<epsilon\>|)>+ln<around*|(|<big|sum><rsub|j><rsup|N<rsub|c>>exp<around*|(|a<rsub|j>|)>|)>,
  </equation*>

  if <math|a<rsub|i>\<less\>a<rsub|min>> and
  <math|\<partial\><with|math-font|cal|L>/\<partial\>a<rsub|i>\<gtr\>0> (i.e.
  wants to decrease itself<\footnote>
    Remind that generally a variable <math|z> decreases iff
    <math|\<partial\><with|math-font|cal|L>/\<partial\>z\<gtr\>0>.
  </footnote>) at some iteration (with a small
  <math|<around*|\||a<rsub|i>-a<rsub|min>|\|>>), then in the next iteration,
  clip <math|\<partial\><with|math-font|cal|L>/\<partial\>a<rsub|i>\<rightarrow\>0>.
  Then the <math|a<rsub|i>> will be ``frozen'' in the next iteration, until
  it wants to increase itself (i.e. when <math|\<partial\><with|math-font|cal|L>/\<partial\>a<rsub|i>\<less\>0>).

  <\problem>
    But the un-frozen <math|a<rsub|j>>s can increase themselves, thus
    increases the <math|a<rsub|min>>. So, if the frozen <math|a<rsub|i>>
    keeps <math|\<partial\><with|math-font|cal|L>/\<partial\>a<rsub|i>\<gtr\>0>,
    then the minimal value of <math|c<rsub|i><around*|(|a|)>> cannot be
    bounded lowerly.
  </problem>

  Comparing with clipping of <math|c<around*|(|a|)>>, clipping of <math|a>
  additionally benefits that it naturally avoids the problem mentioned in the
  section ``Re-scaling of <math|a>'': early random searching makes <math|a>
  dispersed, thus enlarges the elapsed time of convergence after reached the
  targets <math|\<mu\>> and <math|\<zeta\>>. Indeed, by clipping, <math|a>
  becomes centered, even in the early random seaching epoch.

  <subsection|Approximations>

  Comparing to the traditional MAP approach, using multi-peak mixture model
  makes the <math|\<bbb-H\><around*|(|q|)>> complicated, especially in the
  optimization process.

  <subsubsection|Entropy Lower Bound>

  Consider any mixture distribution with p.d.f.
  <math|<big|sum><rsub|i><rsup|N<rsub|c>>c<rsub|i> q<rsub|i>> where <math|c>s
  are the categorical probabilities and <math|q<rsub|i>> the p.d.f. of the
  component distributions of the mixture.

  <\equation*>
    \<bbb-H\><around*|[|<big|sum><rsub|i><rsup|N<rsub|c>>c<rsub|i>
    q<rsub|i>|]>\<geqslant\><big|sum><rsub|i><rsup|N<rsub|c>>c<rsub|i>
    \<bbb-H\><around*|[|q<rsub|i>|]>.
  </equation*>

  So, if define

  <\equation>
    <with|math-font|cal|<with|math-font|cal|L><rprime|'>><around*|(|a,\<mu\>,\<zeta\>|)>\<assign\>-<big|sum><rsub|i><rsup|N<rsub|c>>
    c<rsub|i><around*|(|a|)> <around*|{|\<bbb-E\><rsub|\<Phi\><around*|(|\<eta\>;0,1|)>><around*|[|ln
    p<around*|(|\<sigma\><around*|(|\<zeta\><rsub|i>|)>
    \<eta\>+\<mu\><rsub|i>;D|)>|]>+\<bbb-H\><around*|[|\<Phi\><around*|(|\<theta\>,\<mu\><rsub|i>,\<sigma\><around*|(|\<zeta\><rsub|i>|)>|)>|]>|}>,
  </equation>

  then we have

  <\equation*>
    <with|math-font|cal|L><rprime|'>\<geqslant\><with|math-font|cal|L\<geqslant\>min<around*|(|<with|math-font|cal|L>|)>\<gtr\>-\<infty\>>,
  </equation*>

  thus <math|<with|math-font|cal|L><rprime|'>> has an global minimum. In this
  way, the entropy part becomes completely analytic (and simple).

  However, as experiment on Gaussian mixture model shows, using entropy lower
  bound cannot get the enough accuracy as using entropy does. We will not use
  this approximation.

  <subsection|Stochastic Optimization>

  <subsubsection|Difference between Bayesian and Traditional Methods>

  Suppose, instead of use the whole dataset, we employ mini-batch technique.
  Since all data are independent, if suppose that <math|D<rsub|m>> is
  unbiased in <math|D>, then we have,

  <\equation>
    ln p<around*|(|D\|\<theta\>|)>=<big|sum><rsub|D>p<around*|(|<around*|(|x<rsub|i>,y<rsub|i>,\<sigma\><rsub|i>|)>\|\<theta\>|)>\<approx\><frac|N<rsub|D>|N<rsub|m>><big|sum><rsub|D<rsub|m>>p<around*|(|<around*|(|x<rsub|i>,y<rsub|i>,\<sigma\><rsub|i>|)>\|\<theta\>|)>=<frac|N<rsub|D>|N<rsub|m>>
    ln p<around*|(|D<rsub|m>\|\<theta\>|)>.
  </equation>

  Then,

  <\equation>
    ln p<around*|(|\<theta\>;D|)>=ln p<around*|(|D\|\<theta\>|)>+ln
    p<around*|(|\<theta\>|)>=<frac|N<rsub|D>|N<rsub|m>> ln
    p<around*|(|D<rsub|m>\|\<theta\>|)>+ln p<around*|(|\<theta\>|)>,
  </equation>

  thus as previous

  <\equation>
    ln p<around*|(|\<theta\>;D|)>=<frac|N<rsub|D>|N<rsub|m>><big|sum><rsub|<around*|(|x<rsub|i>,y<rsub|i>,\<sigma\><rsub|i>|)>\<in\>D<rsub|m>><around*|{|-<frac|1|2>ln
    <around*|(|2 \<pi\> \<sigma\><rsub|i><rsup|2>|)>-<frac|1|2>
    <around*|(|<frac|y<rsub|i>-f<around*|(|x<rsub|i>;\<theta\>|)>|\<sigma\><rsub|i>>|)><rsup|2>|}>+ln
    p<around*|(|\<theta\>|)>.
  </equation>

  In this we meet one of the main differences between the Bayesian and the
  traditional. In the traditional method, <math|N<rsub|D>> does not matters
  in training, being absent in the optimizer. However, in Bayesian, the
  number of data that are employed is encoded into Bayesian model, and has
  to, since the greater number of data gives more confidence. So, while using
  stochastic optimization in Bayesian mode, the factor
  <math|N<rsub|D>/N<rsub|m>> of likelihood has to be taken into account. We
  have to know how many data we actually have, thus how confident we are.

  <section|Prediction>

  <\theorem>
    Let <math|p<around*|(|\<theta\>\|D|)>> the posterior, and
    <math|<wide|p|~><around*|(|\<theta\>\|D|)>=C p<around*|(|\<theta\>\|D|)>>
    for arbitrary contant <math|C>, but unknown. For the arbitrary
    ``observable'' <math|y=g<around*|(|x;\<theta\>|)>>, for
    <math|\<forall\>x>, Bayesian inference gives prediction
    <math|<around*|\<langle\>|g|\<rangle\>><around*|(|x|)>> defined as

    <\equation*>
      <around*|\<langle\>|g|\<rangle\>><around*|(|x|)>\<assign\><big|int>\<mathd\>\<theta\>
      p<around*|(|\<theta\>\|D|)> g<around*|(|x;\<theta\>|)>
    </equation*>

    can be computed by the trained distribution
    <math|q<around*|(|\<theta\>;z<rsub|\<ast\>>|)>>
    (<math|z\<assign\><around*|(|a,\<mu\>,\<zeta\>|)>> for short, and star
    notation for representing the trained) by Monte-Carlo integral

    <\equation*>
      <around*|\<langle\>|g|\<rangle\>><around*|(|x|)>\<approx\><big|sum><rsub|i=1><rsup|N<rsub|s>><around*|[|<frac|w<around*|(|\<theta\><rsub|i>;z<rsub|\<ast\>>,D|)>|<big|sum><rsub|j=1><rsup|N<rsub|s>>w<around*|(|\<theta\><rsub|j>;z<rsub|\<ast\>>,D|)>>\<times\>g<around*|(|x;\<theta\><rsub|i>|)>|]>,
    </equation*>

    with

    <\equation*>
      w<around*|(|\<theta\><rsub|i>;z<rsub|\<ast\>>,D|)>\<assign\><frac|<wide|p|~><around*|(|\<theta\><rsub|i>\|D|)>|q<around*|(|\<theta\><rsub|i>;z<rsub|\<ast\>>|)>>,
    </equation*>

    where <math|<around*|{|\<theta\><rsub|i>:i=1,\<ldots\>,N<rsub|s>|}>>
    sampled from <math|q<around*|(|\<theta\>;z<rsub|\<ast\>>|)>>. The error
    of the approximation can be estimated as
    <math|<with|math-font|cal|O><around*|(|1/<sqrt|N<rsub|s>>|)>> and is
    independent of the dimension of the space of <math|\<theta\>>.
  </theorem>

  <\proof>
    Since

    <\equation*>
      <big|int>\<mathd\>\<theta\> <wide|p|~><around*|(|\<theta\>\|D|)>=C\<times\><big|int>\<mathd\>\<theta\>
      p<around*|(|\<theta\>\|D|)>=C\<times\>1
    </equation*>

    and

    <\equation*>
      <big|int>\<mathd\>\<theta\> <wide|p|~><around*|(|\<theta\>\|D|)>=<big|int>\<mathd\>\<theta\>
      q<around*|(|\<theta\>;z<rsub|\<ast\>>|)><frac|<wide|p|~><around*|(|\<theta\>\|D|)>|q<around*|(|\<theta\>;z<rsub|\<ast\>>|)>>\<approx\><big|sum><rsub|i=1><rsup|N<rsub|s>><frac|<wide|p|~><around*|(|\<theta\><rsub|i>\|D|)>|q<around*|(|\<theta\><rsub|i>;z<rsub|\<ast\>>|)>>
    </equation*>

    where <math|<around*|{|\<theta\><rsub|i>:i=1,\<ldots\>,N<rsub|s>|}>>
    sampled from <math|q<around*|(|\<theta\>;z<rsub|\<ast\>>|)>>, we get

    <\equation*>
      C=<big|sum><rsub|i=1><rsup|N<rsub|s>><frac|<wide|p|~><around*|(|\<theta\><rsub|i>\|D|)>|q<around*|(|\<theta\><rsub|i>;z<rsub|\<ast\>>|)>>.
    </equation*>

    Then,

    <\eqnarray*>
      <tformat|<table|<row|<cell|<big|int>\<mathd\>\<theta\>
      p<around*|(|\<theta\>\|D|)> g<around*|(|x;\<theta\>|)>>|<cell|=>|<cell|<frac|1|C>
      <big|int>\<mathd\>\<theta\> <wide|p|~><around*|(|\<theta\>\|D|)>
      g<around*|(|x;\<theta\>|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|C>
      <big|int>\<mathd\>\<theta\> q<around*|(|\<theta\>;z<rsub|\<ast\>>|)><frac|<wide|p|~><around*|(|\<theta\>\|D|)>|q<around*|(|\<theta\>;z<rsub|\<ast\>>|)>>
      g<around*|(|x;\<theta\>|)>>>|<row|<cell|>|<cell|\<approx\>>|<cell|<frac|1|C>\<times\><big|sum><rsub|i=1><rsup|N<rsub|s>><around*|[|<frac|<wide|p|~><around*|(|\<theta\><rsub|i>\|D|)>|q<around*|(|\<theta\><rsub|i>;z<rsub|\<ast\>>|)>>
      g<around*|(|x;\<theta\><rsub|i>|)>|]>.>>>>
    </eqnarray*>

    Denoting

    <\equation*>
      w<around*|(|\<theta\><rsub|i>;z<rsub|\<ast\>>,D|)>\<assign\><frac|<wide|p|~><around*|(|\<theta\><rsub|i>\|D|)>|q<around*|(|\<theta\><rsub|i>;z<rsub|\<ast\>>|)>>,
    </equation*>

    and with some direct re-arrangements, we get

    <\equation*>
      <big|int>\<mathd\>\<theta\> p<around*|(|\<theta\>\|D|)>
      g<around*|(|x;\<theta\>|)>\<approx\><big|sum><rsub|i=1><rsup|N<rsub|s>><around*|[|<frac|w<around*|(|\<theta\><rsub|i>;z<rsub|\<ast\>>,D|)>|<big|sum><rsub|j=1><rsup|N<rsub|s>>w<around*|(|\<theta\><rsub|j>;z<rsub|\<ast\>>,D|)>>\<times\>g<around*|(|x;\<theta\><rsub|i>|)>|]>
    </equation*>

    which is what we want. In this proof, all the approximations come from
    Monte-Carlo integrals, whose error can thus be estimated as
    <math|<with|math-font|cal|O><around*|(|1/<sqrt|N<rsub|s>>|)>> which is
    independent of the dimension of the space of <math|\<theta\>>.
  </proof>

  Notice that we did not compute the mean value
  <math|g<around*|(|x;\<theta\>|)>> directly, as
  <math|\<bbb-E\><rsub|\<theta\>\<sim\>q<around*|(|\<theta\>;a<rsub|\<ast\>>,\<mu\><rsub|\<ast\>>,\<zeta\><rsub|\<ast\>>|)>><around*|[|g<around*|(|x;\<theta\>|)>|]>>,
  which is not accurate enough since <math|q<around*|(|\<theta\>;z<rsub|\<ast\>>|)>>
  is just an approximation to <math|p<around*|(|\<theta\><mid|\|>D|)>>.
  Instead, our implementation does help gain the accurate enough result of
  the Monte-Carlo integral, at the same time avoids the non-convergence of
  MCMC by ``sampling by importance''.

  <section|Deep Learning>

  It cannot solve the vanishing gradient problem of deep neural network,
  since this problem is intrinsic to the posterior of deep neural network.
  Indeed, the posterior has the shape like
  <math|exp<around*|(|-x<rsup|2>/\<sigma\><rsup|2>|)>> with
  <math|\<sigma\>\<rightarrow\>0>, where <math|x> is the variable (argument)
  of the posterior. It has a sharp peak, located at a tiny area, with all
  other region extremely flat. The problem of find this peak, or
  equivalently, finding its tiny area, is intrinsically intractable.

  So, even for Bayesian neural network, a layer by layer abstraction along
  depth cannot be absent.

  <section|Transfer Learning>

  Transfer learning demands that the model can be separated so that, say,
  some lower level layers can be extracted out and directly transfered to
  another model as its lower level layers without any modification on these
  layers. To do so, we have to demand that the marginalization of the
  <math|q<around*|(|\<theta\>;a,\<mu\>,\<zeta\>|)>> on some
  <math|\<theta\><rsub|i>>s shall be easy to take. Indeed, the
  marginalization of our model is straight forward.

  <section|Why not MCMC?>

  Instead, the MCMC approximation to posterior cannot be marginalized easily,
  and even intractable. So, MCMC approximation cannot provide transfer
  learning as we eager. This is the most important reason that we do not
  prefer MCMC. Furthermore, MCMC is not greedy enough so that it converges
  quite slowly, especially in high-dimensional parameter-space.

  <section|Problems>

  <subsection|The Curse of Dimensinality>

  <subsubsection|Range of Sampling>

  Usually, the curse of dimensionality raises in the grid searching or
  numerial integral. And gradient based optimization and Monte Carlo integral
  deal the curse. However, the curse of dimensionality emerges from another
  aspect: the range of sampling of initial values of <math|z> in the
  iteration process of optimization increases as <math|<sqrt|N<rsub|d>>>.

  The large range of sampling then calls for more elapsed time of
  convergence.

  <\example>
    Consider two vector <math|y<rsub|1>> and <math|y<rsub|2>> in
    <math|N<rsub|d>>-dimension Euclidean space
    <math|X<around*|(|N<rsub|d>|)>>. Let <math|Y<rsub|1><around*|(|N<rsub|d>|)>\<assign\><around*|{|x\<in\>X<around*|(|N<rsub|d>|)>:<around*|\<\|\|\>|x-y<rsub|1>|\<\|\|\>>\<less\><around*|\<\|\|\>|x-y<rsub|2>|\<\|\|\>>|}>>.
    Let <math|S<around*|(|N<rsub|d>|)>\<subset\>X<around*|(|N<rsub|d>|)>> as
    the range of sampling. Consider the ratio

    <\equation*>
      R<rsub|1><around*|(|N<rsub|d>|)>\<assign\><frac|<around*|\<\|\|\>|S<around*|(|N<rsub|d>|)>\<cap\>Y<rsub|1><around*|(|N<rsub|d>|)>|\<\|\|\>>|<around*|\<\|\|\>|S<around*|(|N<rsub|d>|)>|\<\|\|\>>>.
    </equation*>

    We find that, e.g. let <math|y<rsub|1>=<around*|(|-1,-1,\<ldots\>,-1|)>>,
    <math|y<rsub|2>=<around*|(|3,3,\<ldots\>,3|)>>, and
    <math|S<around*|(|N<rsub|d>|)>=<around*|(|-r,r|)>\<times\><around*|(|-r,r|)>\<times\>\<cdots\>\<times\><around*|(|-r,r|)>>
    wherein <math|r=10>, <math|R<rsub|1>> becomes unit after
    <math|N<rsub|d>\<geqslant\>?>, and that <math|R<rsub|1>> will be around
    <math|0.5> if let <math|r\<sim\><sqrt|N<rsub|d>>>.<\footnote>
      C.f. the code ``<shell|/docs/curse_of_dimensionality/curse_of_dimensionality.py>''.
    </footnote>
  </example>

  <subsubsection|Relative Error of Inference>

  Theorem <reference|theorem: Relation between Relative Error of Inference
  and Loss> hints that the relative error of inference positively related
  with dimension, as the loss does so.

  <section|Drafts>

  <math|f<around*|(|x;\<theta\>|)>>, <math|D=<around*|{|<around*|(|x<rsub|i>,y<rsub|i>,z<rsub|i>|)>:i=1,2,\<ldots\>,N<rsub|D>|}>>.
  <math|y<rsub|i>=f<around*|(|x<rsub|i>;\<theta\>|)>+\<epsilon\>>,
  <math|\<epsilon\>\<sim\>P<around*|(|0,z<rsub|i>|)>>.

  For instance, suppose <math|P=<with|math-font|cal|N>>, then we have

  <\equation*>
    ln p<around*|(|y<rsub|1:N<rsub|D>>\|x<rsub|1:N<rsub|D>>;\<theta\>|)>=<big|sum><rsub|i><rsup|N<rsub|D>>ln
    p<around*|(|y<rsub|i>\|x<rsub|i>;\<theta\>|)>.
  </equation*>

  From <math|y<rsub|i>=f<around*|(|x<rsub|i>;\<theta\>|)>+\<epsilon\>> gets
  <math|y<rsub|i>-f<around*|(|x<rsub|i>;\<theta\>|)>\<sim\>P<around*|(|0,z<rsub|i>|)>=<with|math-font|cal|N<around*|(|0,z<rsub|i>|)>>>.

  <\equation*>
    <big|sum><rsup|N<rsub|D>><rsub|i>ln p<around*|(|y<rsub|i>\|x<rsub|i>;\<theta\>|)>=-<big|sum><rsub|i><rsup|N<rsub|D>><around*|{|ln<around*|(|2
    <sqrt|z<rsub|i>>|)>+<frac|<around*|(|y<rsub|i>-f<around*|(|x<rsub|i>;\<theta\>|)>|)><rsup|2>|2
    z<rsub|i><rsup|2>>|}>.
  </equation*>
</body>

<\initial>
  <\collection>
    <associate|font-base-size|10>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|2.4|2>>
    <associate|auto-11|<tuple|2.5|3>>
    <associate|auto-12|<tuple|2.6|3>>
    <associate|auto-13|<tuple|3|3>>
    <associate|auto-14|<tuple|3.1|4>>
    <associate|auto-15|<tuple|3.1.1|4>>
    <associate|auto-16|<tuple|3.2|4>>
    <associate|auto-17|<tuple|3.2.1|5>>
    <associate|auto-18|<tuple|3.2.2|5>>
    <associate|auto-19|<tuple|3.2.3|5>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-20|<tuple|3.3|5>>
    <associate|auto-21|<tuple|3.3.1|5>>
    <associate|auto-22|<tuple|3.4|6>>
    <associate|auto-23|<tuple|3.4.1|6>>
    <associate|auto-24|<tuple|4|6>>
    <associate|auto-25|<tuple|5|6>>
    <associate|auto-26|<tuple|6|6>>
    <associate|auto-27|<tuple|7|6>>
    <associate|auto-28|<tuple|8|?>>
    <associate|auto-29|<tuple|8.1|?>>
    <associate|auto-3|<tuple|2.1|1>>
    <associate|auto-30|<tuple|8.1.1|?>>
    <associate|auto-31|<tuple|8.1.2|?>>
    <associate|auto-32|<tuple|9|?>>
    <associate|auto-33|<tuple|9|?>>
    <associate|auto-34|<tuple|8.1.2|?>>
    <associate|auto-35|<tuple|9|?>>
    <associate|auto-4|<tuple|2.1.1|1>>
    <associate|auto-5|<tuple|2.2|2>>
    <associate|auto-6|<tuple|2.2.1|2>>
    <associate|auto-7|<tuple|2.2.2|2>>
    <associate|auto-8|<tuple|2.2.3|2>>
    <associate|auto-9|<tuple|2.3|2>>
    <associate|figure: 1|<tuple|1|4>>
    <associate|footnote-1|<tuple|1|1>>
    <associate|footnote-10|<tuple|10|?>>
    <associate|footnote-2|<tuple|2|1>>
    <associate|footnote-3|<tuple|3|2>>
    <associate|footnote-4|<tuple|4|3>>
    <associate|footnote-5|<tuple|5|4>>
    <associate|footnote-6|<tuple|6|4>>
    <associate|footnote-7|<tuple|7|?>>
    <associate|footnote-8|<tuple|8|?>>
    <associate|footnote-9|<tuple|9|?>>
    <associate|footnr-1|<tuple|1|1>>
    <associate|footnr-10|<tuple|10|?>>
    <associate|footnr-2|<tuple|2|1>>
    <associate|footnr-3|<tuple|3|2>>
    <associate|footnr-4|<tuple|4|3>>
    <associate|footnr-5|<tuple|5|4>>
    <associate|footnr-6|<tuple|6|4>>
    <associate|footnr-7|<tuple|7|?>>
    <associate|footnr-8|<tuple|8|?>>
    <associate|footnr-9|<tuple|9|?>>
    <associate|theorem: Relation between Relative Error of Inference and
    Loss|<tuple|1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Notations>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Neural
      Network for Posterior (nn4post)> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>The Model
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|2tab>|2.1.1<space|2spc>Numerical Consideration
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Interpretation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|2tab>|2.2.1<space|2spc>As a Mixture Distribution
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|2tab>|2.2.2<space|2spc>As a Generalization
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <with|par-left|<quote|2tab>|2.2.3<space|2spc>As a Neural Network
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>>

      <with|par-left|<quote|1tab>|2.3<space|2spc>Marginalization
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9>>

      <with|par-left|<quote|1tab>|2.4<space|2spc>Loss-Function
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10>>

      <with|par-left|<quote|1tab>|2.5<space|2spc>Relation with the
      Traditional MAP Loss <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11>>

      <with|par-left|<quote|1tab>|2.6<space|2spc>Relation between Relative
      Error of Inference and Loss <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-12>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Optimization>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-13><vspace|0.5fn>

      <with|par-left|<quote|1tab>|3.1<space|2spc>ADVI
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-14>>

      <with|par-left|<quote|2tab>|3.1.1<space|2spc>Derivation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-15>>

      <with|par-left|<quote|1tab>|3.2<space|2spc>Redefination of Gradients
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-16>>

      <with|par-left|<quote|2tab>|3.2.1<space|2spc>Gauge Fixing
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-17>>

      <with|par-left|<quote|2tab>|3.2.2<space|2spc>Re-scaling of
      <with|mode|<quote|math>|a> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-18>>

      <with|par-left|<quote|2tab>|3.2.3<space|2spc>Frozen-out Problem
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-19>>

      <with|par-left|<quote|1tab>|3.3<space|2spc>Approximations
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-20>>

      <with|par-left|<quote|2tab>|3.3.1<space|2spc>Entropy Lower Bound
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-21>>

      <with|par-left|<quote|1tab>|3.4<space|2spc>Stochastic Optimization
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-22>>

      <with|par-left|<quote|2tab>|3.4.1<space|2spc>Difference between
      Bayesian and Traditional Methods <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-23>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Prediction>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-24><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Deep
      Learning> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-25><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|6<space|2spc>Transfer
      Learning> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-26><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|7<space|2spc>Why
      not MCMC?> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-27><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|8<space|2spc>Problems>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-28><vspace|0.5fn>

      <with|par-left|<quote|1tab>|8.1<space|2spc>The Curse of Dimensinality
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-29>>

      <with|par-left|<quote|2tab>|8.1.1<space|2spc>Range of Sampling
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-30>>

      <with|par-left|<quote|2tab>|8.1.2<space|2spc>Relative Error of
      Inference <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-31>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|9<space|2spc>Drafts>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-32><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>