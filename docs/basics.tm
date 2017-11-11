<TeXmacs|1.99.1>

<style|generic>

<\body>
  <section|Gradient Decent Optimization>

  <subsection|Illustration>

  At one iteration, for each variable, <with|color|red|while fixing all other
  variables at their temperal values>, increases or decrease by a small
  step-length so that the loss can be decreased. And for each variable, this
  step-length is determined by how much can it decrease the loss: the more
  (decrement) the longer (step-length), so that the efficiency can be
  maximized.

  <section|Bayesian>

  <subsection|Prior-Posterior Iteration of Encoding>

  Bayesian approach is an iterative process of encoding. It is encoding since
  it encodes the prior assumptions based on human illustration and the
  likelihood based on data. And it is iterative since when adding new data,
  the prior of the of the new posterior is the posterior gained by the old
  data.

  <subsection|Comparing to MAP>

  Traditional MAP approach gains a best fit point only on parameter-space,
  and being irrelavent to the size of dataset. Contrarily, Bayesian approach
  gains a distribution, which encodes the human prior, data character, and
  importantly the size of data, encoded as the confidence level in the
  posterior. So, Bayesian approach extract more information from data.

  <subsection|Overcoming Over-Fitting>

  This is one aspect of how over-fitting is overcome in Bayesian approach.
  When the size of dataset is small, the confidence level decreases, the
  sampling in parameter-space becomes dispersing; and when the size is large,
  the confidence level increases, the sampling in parameters becomes
  concentrated. In one word, the uncertainty caused by the lack of data is
  explicitly reflected in Bayesian approach, as it should be. (However, this
  uncertainty is absent in traditional MAP approach.)

  Another aspect of how over-fitting is overcome in Bayesian approach is that
  it can encodes the higher but not the highest peaks in posterior. That is,
  encoding more information of data, which, even though not being the most
  important, can still be significant for inference. Contrarily, the
  traditional MAP approach can only encode the highest peak in posterior.
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?|../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-2|<tuple|1.1|?|../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-3|<tuple|2|?|../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-4|<tuple|2.1|?|../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-5|<tuple|2.2|?|../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-6|<tuple|2.3|?|../../../.TeXmacs/texts/scratch/no_name_1.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Bayesian>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Prior-Posterior Iteration of
      Encoding <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|1.2<space|2spc>Comparing to MAP
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|1.3<space|2spc>Over-Fitting
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>
    </associate>
  </collection>
</auxiliary>