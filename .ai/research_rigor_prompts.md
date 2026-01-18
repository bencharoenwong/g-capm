# AI Prompts for Statistical Rigor in Quant Research

This library contains prompts to help you use AI as a research assistant while maintaining statistical rigor.

**Philosophy**: AI is a tool to help you think critically, not to replace critical thinking.

---

## Category 1: Assumption Testing

### Prompt: Testing Time Series Assumptions

```
I'm about to run a regression with time series data.

Data:
- Dependent variable: [DESCRIBE]
- Independent variables: [DESCRIBE]
- Frequency: [daily/monthly/etc.]
- Period: [dates]

Before I run the regression, what assumptions should I test?

Please provide:
1. List of critical assumptions for time series regression
2. Specific tests to run (with Python code if possible)
3. How to interpret the results
4. What to do if assumptions are violated

Be rigorous - I'm writing this for a research paper, not a blog post.
```

**Example conversation:**

```
User: I'm about to run a CAPM regression with daily Bitcoin returns vs S&P 500.

Data:
- Dependent: Bitcoin daily returns (2016-2021)
- Independent: S&P 500 daily returns
- Frequency: Daily
- Period: 2016-03-10 to 2021-08-30

Before I run the regression, what assumptions should I test?
```

**AI should mention:**
- Stationarity (ADF test)
- Heteroscedasticity (White test, ARCH effects)
- Autocorrelation (Ljung-Box test, Durbin-Watson)
- Normality (for inference, though robust to violations with large N)
- Linear relationship
- No structural breaks

---

### Prompt: Interpreting Stationarity Tests

```
I ran an Augmented Dickey-Fuller test and got these results:

ADF Statistic: [VALUE]
P-value: [VALUE]
Critical values: 1%: [X], 5%: [Y], 10%: [Z]

Questions:
1. Is this series stationary?
2. How confident can I be?
3. What should I do if it's non-stationary?
4. Are there alternative tests I should run?
5. What are the implications for my regression analysis?

Context: I'm testing [DESCRIBE YOUR SERIES] for a CAPM/factor model analysis.
```

---

## Category 2: Lookahead Bias Detection

### Prompt: Checking for Lookahead Bias

```
Review this backtesting code for lookahead bias:

```python
[YOUR CODE HERE]
```

Please:
1. Identify any lookahead bias
2. Explain why each instance is problematic
3. Show how to fix it
4. Provide corrected code

Be thorough - lookahead bias is subtle and can hide in unexpected places.

Key questions to check:
- Am I using return at time t to make decision at time t?
- Did I shift signals appropriately?
- Is training/test split temporal (no shuffling)?
- Am I using any "realized" variables that wouldn't be known at decision time?
```

**Example:**

```
User: Review this backtesting code for lookahead bias:

```python
def backtest(data):
    data['ma_20'] = data['close'].rolling(20).mean()
    data['signal'] = (data['close'] > data['ma_20']).astype(int)
    data['strategy_return'] = data['signal'] * data['returns']
    return data['strategy_return'].sum()
```

AI should catch:
- Signal at time t uses close at time t
- Should shift signal forward
- Should use shifted MA or calculate with lag
```

---

### Prompt: Walk-Forward Validation Design

```
I want to implement walk-forward validation for a trading strategy.

Strategy details:
- Parameters to optimize: [LIST PARAMETERS]
- Data: [FREQUENCY], [PERIOD]
- Typical parameter search: [DESCRIBE RANGE]

Please help me design a proper walk-forward framework:
1. Suggest training/validation/test window sizes
2. Help me avoid lookahead bias in parameter optimization
3. Show how to handle expanding vs rolling windows
4. Explain how to aggregate results across folds
5. Provide Python implementation template

Context: This is for a research paper, so it needs to be methodologically sound.
```

---

## Category 3: Multiple Testing Corrections

### Prompt: Multiple Testing Correction

```
I'm testing multiple hypotheses in my research.

Number of tests: [N]
Type of tests: [independent/correlated]
Significance level desired: [usually 0.05]

Scenarios:
1. Testing N different assets for positive alpha
2. Testing M different factor specifications
3. Trying K different parameter combinations

Questions:
1. What's the appropriate correction method?
2. How do I calculate the corrected p-value threshold?
3. Should I use Bonferroni, Holm-Bonferroni, or FDR?
4. How do I report this in my results?

Please explain the trade-offs between methods and recommend the most appropriate for my use case.
```

**Example:**

```
User: I'm testing 20 different cryptocurrencies for positive CAPM alpha.

Number of tests: 20
Type: Independent tests (different assets)
Significance level: 0.05

What correction should I use and what's my new p-value threshold?
```

**AI should explain:**
- Bonferroni: p < 0.05/20 = 0.0025
- Holm-Bonferroni: More powerful, step-down procedure
- FDR (Benjamini-Hochberg): If willing to accept some false discoveries
- How to report: "After Bonferroni correction for 20 tests..."

---

## Category 4: Bias Awareness

### Prompt: Survivorship Bias Check

```
Analyze my dataset for survivorship bias:

Dataset description:
- [DESCRIBE YOUR DATA]
- Time period: [DATES]
- Assets included: [HOW SELECTED]
- Data source: [WHERE FROM]

Questions:
1. Is survivorship bias likely present?
2. How might it affect my results?
3. Can I quantify the potential bias?
4. How should I acknowledge this limitation?
5. Are there alternative datasets without this bias?

I want to be honest about limitations in my research paper.
```

---

### Prompt: Transaction Cost Modeling

```
Help me model realistic transaction costs for my strategy.

Strategy details:
- Asset class: [stocks/crypto/futures]
- Trade frequency: [daily/weekly/monthly]
- Typical position size: [DOLLAR AMOUNT or % of portfolio]
- Time period: [DATES]

Please provide:
1. Realistic transaction cost assumptions (in bps)
2. How to model bid-ask spread
3. How to account for market impact
4. Python code to apply costs to backtest
5. Sensitivity analysis framework (test different cost levels)

I need this to be realistic - not overly optimistic.
```

---

## Category 5: Specification Robustness

### Prompt: Robustness Check Design

```
I found a significant result in my analysis. Help me design robustness checks.

Main result:
- [DESCRIBE YOUR FINDING]
- Specification used: [MODEL/METHOD]
- Sample: [DATA DESCRIPTION]

Please suggest:
1. At least 5 robustness checks I should run
2. What would invalidate my result?
3. How to present robustness results
4. What to do if robustness checks fail?

I want to be thorough before claiming this as a real finding.
```

**Example:**

```
User: I found that Bitcoin has significant positive alpha.

Main result:
- Alpha = 0.31% per day (p = 0.02)
- Specification: CAPM regression, daily data, 2016-2021
- Sample: Bitcoin vs S&P 500 returns

Suggest robustness checks.
```

**AI should suggest:**
- Different time periods (exclude COVID, split pre/post 2020)
- Different frequencies (monthly instead of daily)
- Different market proxies (Total Market vs S&P 500)
- Subperiod analysis
- Control for size/value/momentum factors
- Check for ARCH effects and use robust SEs

---

## Category 6: Result Interpretation

### Prompt: Statistical vs Economic Significance

```
I got these regression results:

Coefficient: [VALUE]
Standard error: [VALUE]
P-value: [VALUE]
R-squared: [VALUE]
Sample size: [N]

Context: [WHAT YOU'RE TESTING]

Questions:
1. Is this statistically significant?
2. Is this economically significant?
3. What's a reasonable effect size for this context?
4. Could this be significant just because of large sample size?
5. How should I interpret and report this?

Help me think critically about what this result means.
```

---

### Prompt: Validating AI-Generated Code

```
You helped me write this statistical code. Now I need to validate it's correct.

Code:
```python
[PASTE CODE]
```

Please:
1. Verify the statistical methodology is sound
2. Check for common implementation errors
3. Suggest how I can test if it's working correctly
4. Provide a simple example where I know the answer
5. Point out any edge cases I should handle

I don't want to publish incorrect results because of a coding error.
```

---

## Category 7: Research Design

### Prompt: Pre-Registration Design

```
I want to pre-register my analysis to avoid p-hacking.

Research question: [YOUR QUESTION]
Available data: [DESCRIBE]

Help me write a complete pre-registration that includes:
1. Specific hypothesis (H0 and H1)
2. Exact model specification
3. Variable definitions
4. Sample selection criteria
5. Statistical tests planned
6. Significance level
7. Planned robustness checks
8. What would constitute rejecting H0?

Make this specific enough that I can't deviate later.
```

---

## Category 8: Critical Evaluation

### Prompt: Critique My Analysis

```
I want you to be VERY CRITICAL of my analysis. Don't be nice.

My analysis:
[DESCRIBE WHAT YOU DID]

My results:
[DESCRIBE FINDINGS]

Please:
1. Identify every potential flaw
2. Suggest alternative explanations for my results
3. Point out any unjustified assumptions
4. Find the weakest part of my methodology
5. Tell me what a skeptical reviewer would say
6. Rate likelihood this result is real vs artifact (%)

Be harsh - I'd rather find problems now than after publishing.
```

---

## Category 9: Reporting Standards

### Prompt: Academic Reporting Template

```
Help me write up my results following academic standards.

My analysis:
- Method: [DESCRIBE]
- Results: [SUMMARIZE]
- Sample: [DESCRIBE]

Please provide a template that includes:
1. How to report the main result (with SE, CI, p-value)
2. How to present robustness checks
3. How to discuss limitations
4. How to report out-of-sample performance
5. What to include in tables/figures
6. How to handle non-significant results

I'm writing for a finance journal, so precision matters.
```

---

## Category 10: Methodology Clarification

### Prompt: Understand Statistical Concept

```
I need to understand [CONCEPT] for my research.

My current understanding: [WHAT YOU THINK YOU KNOW]

What I'm confused about: [SPECIFIC CONFUSION]

Context of use: [WHY YOU NEED TO KNOW THIS]

Please:
1. Explain the concept rigorously (not just intuition)
2. Show the mathematical formulation
3. Explain when to use vs not use it
4. Give an example from finance
5. Point out common mistakes
6. Recommend authoritative references

I need proper understanding, not just enough to use a function.
```

**Example:**

```
User: I need to understand heteroscedasticity for my CAPM regression.

Current understanding: Variance isn't constant over time

Confused about:
- When is this a problem vs just annoying?
- White standard errors vs GARCH - which to use when?
- Does it bias my beta estimate or just the standard errors?

Context: Daily stock return regressions for academic research
```

---

## Meta-Prompt: Research Advisor

```
Act as my research advisor for a quantitative finance project.

Project: [DESCRIBE]

I'm at the stage of: [EXPLORATORY/ANALYSIS/WRITING UP]

Current concerns:
1. [CONCERN 1]
2. [CONCERN 2]

Please:
1. Ask me clarifying questions about my methodology
2. Identify potential flaws I haven't considered
3. Suggest additional analyses that would strengthen results
4. Point out any violations of best practices
5. Recommend how to proceed

Be rigorous and don't let me cut corners.
```

---

## How to Use These Prompts

### 1. Customize for Your Case
- Replace [PLACEHOLDERS] with your specific details
- Add relevant context
- Be specific about your data and methods

### 2. Iterate
- AI's first answer may not be complete
- Ask follow-up questions
- Request clarification or deeper explanation

### 3. Verify
- **Never trust AI blindly on statistical methodology**
- Cross-reference with textbooks (Greene, Hamilton, etc.)
- Test on simulated data where you know the answer
- Ask AI to cite sources

### 4. Document
- Save useful AI conversations
- Note what you learned
- Build your own prompt refinements

---

## Warning Signs (When AI Might Be Wrong)

Be extra skeptical if AI:
- Gives very simple answer to complex statistical question
- Doesn't mention assumptions or limitations
- Suggests methods without explaining when they apply
- Contradicts authoritative sources
- Seems overly confident about nuanced topics

**When in doubt:**
1. Ask AI to explain the reasoning
2. Request citations
3. Test on simple example
4. Consult textbook/papers
5. Ask multiple AIs and compare

---

## Good Research Habits with AI

✓ **Do:**
- Use AI to learn statistical concepts
- Ask AI to review your methodology
- Have AI suggest robustness checks
- Use AI to generate test cases
- Ask AI to critique your work

✗ **Don't:**
- Let AI choose your statistical tests without understanding why
- Trust AI's numerical calculations without verification
- Use AI-generated code without understanding it
- Skip learning fundamentals because "AI can do it"
- Ignore AI when it points out flaws

---

## Advanced: Adversarial AI Use

**Technique**: Use AI to attack your own research

```
I'm going to defend my research. You play the role of a very skeptical reviewer.

My claim: [YOUR RESULT]
My methodology: [YOUR METHODS]

Your job:
- Find every possible flaw
- Suggest alternative explanations
- Question my assumptions
- Point out omitted variables
- Identify potential biases
- Challenge my interpretations

Be harsh. I want to strengthen my work, not hear it's good.
```

This helps you find weaknesses before reviewers do!

---

## Resources

**Textbooks to Cross-Reference:**
- Greene: Econometric Analysis
- Hamilton: Time Series Analysis
- Cochrane: Asset Pricing
- Campbell, Lo, MacKinlay: Econometrics of Financial Markets

**When AI conflicts with these sources: Trust the textbook.**

---

## Contributing

Found a useful prompt? Add it to this library!

Good prompts are:
- Specific enough to be useful
- General enough to be reusable
- Emphasize critical thinking
- Request rigorous methodology
- Encourage verification
