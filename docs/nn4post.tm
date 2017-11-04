<TeXmacs|1.99.1>

<style|generic>

<\body>
  <section|Notations>

  <subsection|Model & Data>

  Let <math|f<around*|(|x;\<theta\>|)>> a function of <math|x> with parameter
  <math|\<theta\>>. Let <math|y=f<around*|(|x;\<theta\>|)>> an observable,
  thus the observed value obeys a Gaussian distribution. Let <math|D> denotes
  a set of observations, <math|D\<assign\><around*|{|<around*|(|x<rsub|i>,y<rsub|i>,\<sigma\><rsub|i>|)>:i=1,\<ldots\>,N<rsub|D>|}>>,
  wherein <math|x<rsub|i>> is the <math|i>th input, <math|y<rsub|i>> its
  observed value, and <math|\<sigma\><rsub|i>> the observational error of
  <math|y<rsub|i>>. We may employ mini-batch technique, thus denote
  <math|D<rsub|m>\<assign\><around*|{|<around*|(|x<rsub|i>,y<rsub|i>,\<sigma\><rsub|i>|)>:i=1,\<ldots\>,N<rsub|m>|}>\<subset\>D>
  as a mini-batch, with batch-size <math|N<rsub|m>\<leqslant\>N<rsub|D>>. We
  use <math|\<bbb-E\><rsub|f<around*|(|\<theta\>|)>><around*|[|g<around*|(|\<theta\>|)>|]>>
  represent the expectation of function <math|g> of a random variable obeys
  the p.d.f. <math|f>. <math|\<Phi\>> is for Gaussian p.d.f..

  <section|Bayesian>

  <subsection|Prior-Posterior Iteration>

  <subsection|Bayesian as Information Encoder>

  Comparing with the traditional method, what is the advantage of Bayesian
  way? The answer is, it encodes more information of data into model. Indeed,
  it does not encodes the value of peak of the posterior only, as traditional
  method does, but also much more information on the posterior. XXX

  <section|Neural Network for Posterior (nn4post)>

  <subsection|The Model>

  Suppose we have some prior on <math|\<theta\>>,
  <math|p<around*|(|\<theta\>|)>>, we gain the unormalized posterior
  <math|p<around*|(|D\|\<theta\>|)> p<around*|(|\<theta\>|)>>. With <math|D>
  arbitrarily given, this unormalized posterior is a function of
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
  \<alpha\>>\<in\>\<bbb-R\>> for <math|\<forall\>i,\<forall\>\<alpha\>> and

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
  the fitting is then defined as <math|<with|math-font|cal|L>=-ELBO>, i.e.

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

  <subsection|Redefination of <math|\<partial\><with|math-font|cal|L>/\<partial\>a>>

  Let <math|\<Delta\>r> the learning-rate. Then the updation of
  <math|a<rsub|i>> at one iteration by gradient decent method is

  <\equation*>
    \<Delta\>a<rsub|i>=<frac|\<partial\><with|math-font|cal|L>|\<partial\>a<rsub|i>><around*|(|a,\<mu\>,\<zeta\>|)>
    \<Delta\>r.
  </equation*>

  Notice that redefine the <math|\<partial\><with|math-font|cal|L>/\<partial\>a>
  by

  <\equation*>
    <frac|\<partial\><with|math-font|cal|L>|\<partial\>a<rsub|i>><around*|(|a,\<mu\>,\<zeta\>|)>\<rightarrow\><frac|\<partial\><with|math-font|cal|L>|\<partial\>a<rsub|i>><around*|(|a,\<mu\>,\<zeta\>|)>+C,
  </equation*>

  where <math|C> can be any constant, leaves the updation of
  <math|c<rsub|i><around*|(|a|)>> invariant, since it makes

  <\equation*>
    \<Delta\>a<rsub|i>\<rightarrow\><frac|\<partial\><with|math-font|cal|L>|\<partial\>a<rsub|i>><around*|(|a,\<mu\>,\<zeta\>|)>
    \<Delta\>r+C \<Delta\>r,
  </equation*>

  thus

  <\equation*>
    c<rsub|i><around*|(|a+\<Delta\>a|)>\<rightarrow\><frac|exp<around*|(|a<rsub|i>+\<Delta\>a<rsub|i>+C
    \<Delta\>r|)>|<big|sum><rsub|j>exp<around*|(|a<rsub|j>+\<Delta\>a<rsub|j>+C
    \<Delta\>r|)>>=<frac|exp<around*|(|a<rsub|i>+\<Delta\>a<rsub|i>|)>|<big|sum><rsub|j>exp<around*|(|a<rsub|j>+\<Delta\>a<rsub|j>|)>>=c<rsub|i><around*|(|a+\<Delta\>a|)>.
  </equation*>

  This <math|C> thus provides an additional dof. We can tune the value of
  <math|C> so that the updation of <math|a<rsub|i>> is numerically stable.
  Indeed, let <math|C> be the average of <math|<around*|{|\<partial\><with|math-font|cal|L>/\<partial\>a<rsub|i>:i=1,\<ldots\>,N<rsub|c>|}>>,
  we find a pretty stability of <math|a> as well as a pretty accuracy of
  <math|c> in the iteration process of optimization.

  <subsubsection|Re-scaling of <math|a>>

  \;

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

  <subsubsection|Application>

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\><with|math-font|cal|<with|math-font|cal|L><rprime|'>>|\<partial\>a<rsub|i>><around*|(|a,\<mu\>,\<zeta\>|)>>|<cell|=>|<cell|-c<rsub|i><around*|(|a|)>
    <around*|{|\<bbb-E\><rsub|\<Phi\><around*|(|\<theta\>;\<mu\><rsub|i>,\<sigma\><around*|(|\<zeta\><rsub|i>|)>|)>><around*|[|ln
    p<around*|(|\<theta\>;D|)>|]>+\<bbb-H\><around*|[|\<Phi\><around*|(|\<theta\>,\<mu\><rsub|i>,\<sigma\><around*|(|\<zeta\><rsub|i>|)>|)>|]>+<with|math-font|cal|L><rprime|'><around*|(|a,\<mu\>,\<zeta\>|)>|}>;<eq-number>>>|<row|<cell|<frac|\<partial\><with|math-font|cal|L><rprime|'>|\<partial\>\<mu\><rsub|i\<alpha\>>><around*|(|a,\<mu\>,\<zeta\>|)>>|<cell|=>|<cell|<frac|\<partial\><with|math-font|cal|L><rsub|p>|\<partial\>\<mu\><rsub|i
    \<alpha\>>><around*|(|a,\<mu\>,\<zeta\>|)>+0;<eq-number>>>|<row|<cell|<frac|\<partial\><with|math-font|cal|L><rprime|'>|\<partial\>\<zeta\><rsub|i\<alpha\>>><around*|(|a,\<mu\>,\<zeta\>|)>>|<cell|=>|<cell|\<ldots\><eq-number>>>>>
  </eqnarray*>

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
  have to know how many data we actrually have, thus how confident we are.

  <section|Deep Learning>

  It cannot solve the vanishing gradient problem of deep neural network,
  since this problem is intrinsic to the posterior of deep neural network.
  Indeed, the posterior has the shape like
  <math|exp<around*|(|-x<rsup|2>/\<sigma\><rsup|2>|)>> with
  <math|\<sigma\>\<rightarrow\>0>, where <math|x> is the variable (argument)
  of the posterior. It has a sharp peak, located at a tiny area, with all
  other region extremely flat. The problem of find this peak, or
  equivalently, findng its tiny area, is intrinsically intactable.

  So, even for Bayesian neural network, a layer by layer abstraction along
  depth cannot be absent.

  <section|Transfer Learning>

  <section|Why not MCMC?>
</body>

<\initial>
  <\collection>
    <associate|font-base-size|10>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|3.2.1|2>>
    <associate|auto-11|<tuple|3.2.2|2>>
    <associate|auto-12|<tuple|3.2.3|2>>
    <associate|auto-13|<tuple|3.3|2>>
    <associate|auto-14|<tuple|3.4|2>>
    <associate|auto-15|<tuple|4|3>>
    <associate|auto-16|<tuple|4.1|3>>
    <associate|auto-17|<tuple|4.1.1|3>>
    <associate|auto-18|<tuple|4.2|4>>
    <associate|auto-19|<tuple|4.2.1|4>>
    <associate|auto-2|<tuple|1.1|1>>
    <associate|auto-20|<tuple|4.3|4>>
    <associate|auto-21|<tuple|4.3.1|4>>
    <associate|auto-22|<tuple|4.3.2|5>>
    <associate|auto-23|<tuple|4.4|5>>
    <associate|auto-24|<tuple|4.4.1|5>>
    <associate|auto-25|<tuple|5|5>>
    <associate|auto-26|<tuple|6|6>>
    <associate|auto-27|<tuple|7|6>>
    <associate|auto-28|<tuple|9|?>>
    <associate|auto-29|<tuple|10|?>>
    <associate|auto-3|<tuple|2|1>>
    <associate|auto-30|<tuple|10|?>>
    <associate|auto-31|<tuple|11|?>>
    <associate|auto-32|<tuple|11|?>>
    <associate|auto-33|<tuple|9|?>>
    <associate|auto-34|<tuple|8.1.2|?>>
    <associate|auto-35|<tuple|9|?>>
    <associate|auto-4|<tuple|2.1|1>>
    <associate|auto-5|<tuple|2.2|1>>
    <associate|auto-6|<tuple|3|1>>
    <associate|auto-7|<tuple|3.1|1>>
    <associate|auto-8|<tuple|3.1.1|2>>
    <associate|auto-9|<tuple|3.2|2>>
    <associate|figure: 1|<tuple|1|4>>
    <associate|footnote-1|<tuple|1|1>>
    <associate|footnote-2|<tuple|2|1>>
    <associate|footnote-3|<tuple|3|2>>
    <associate|footnote-4|<tuple|4|3>>
    <associate|footnote-5|<tuple|5|?>>
    <associate|footnr-1|<tuple|1|1>>
    <associate|footnr-2|<tuple|2|1>>
    <associate|footnr-3|<tuple|3|2>>
    <associate|footnr-4|<tuple|4|3>>
    <associate|footnr-5|<tuple|5|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Notations>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Model & Data
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Bayesian>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Prior-Posterior Iteration
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Bayesian as Information
      Encoder <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Neural
      Network for Posterior (nn4post)> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <with|par-left|<quote|1tab>|3.1<space|2spc>The Model
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <with|par-left|<quote|2tab>|3.1.1<space|2spc>Numerical Consideration
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>>

      <with|par-left|<quote|1tab>|3.2<space|2spc>Interpretation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9>>

      <with|par-left|<quote|2tab>|3.2.1<space|2spc>As a Mixture Distribution
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10>>

      <with|par-left|<quote|2tab>|3.2.2<space|2spc>As a Generalization
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11>>

      <with|par-left|<quote|2tab>|3.2.3<space|2spc>As a Neural Network
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-12>>

      <with|par-left|<quote|1tab>|3.3<space|2spc>Marginalization
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-13>>

      <with|par-left|<quote|1tab>|3.4<space|2spc>Loss-Function
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-14>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Optimization>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-15><vspace|0.5fn>

      <with|par-left|<quote|1tab>|4.1<space|2spc>ADVI
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-16>>

      <with|par-left|<quote|2tab>|4.1.1<space|2spc>Derivation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-17>>

      <with|par-left|<quote|1tab>|4.2<space|2spc>Redefination of
      <with|mode|<quote|math>|\<partial\><with|math-font|<quote|cal>|L>/\<partial\>a>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-18>>

      <with|par-left|<quote|2tab>|4.2.1<space|2spc>Re-scaling of
      <with|mode|<quote|math>|a> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-19>>

      <with|par-left|<quote|1tab>|4.3<space|2spc>Approximations
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-20>>

      <with|par-left|<quote|2tab>|4.3.1<space|2spc>Entropy Lower Bound
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-21>>

      <with|par-left|<quote|2tab>|4.3.2<space|2spc>Application
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-22>>

      <with|par-left|<quote|1tab>|4.4<space|2spc>Stochastic Optimization
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-23>>

      <with|par-left|<quote|2tab>|4.4.1<space|2spc>Difference between
      Bayesian and Traditional Methods <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-24>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Deep
      Learning> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-25><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|6<space|2spc>Transfer
      Learning> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-26><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|7<space|2spc>Why
      not MCMC?> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-27><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>