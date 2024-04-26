# Notes on input data and availability of sources

## Notes from the VoC paper and Welch and Goyal (WG) paper:

Legend: ✔️ - available from Welch and Goyal (2022) data, ❌ - not available, ❓ - not sure, ❗ - potential problems (e.g. look-ahead bias).

[VoC]: 
- Target `R`: monthly excess return of the CRSP value-weighted index (`R = CRSP_SPvw - tbl` or `-Rfree`. [WG]: The risk-free rate from 1920 to 2005 is the Treasury-bill rate. ❓ `Rfree` and `tbl` columns, after annualizing `tbl` do not match but they are very close to each other, where `tbl` appears smoother (see plots below). I used `tbl` for the risk-free rate.
- 15 predictor variables from Welch and Goyal (2008) available monthly over the sample from 1926 to 2020.

[WG]: Our dependent variable is always the equity premium, that is, the total rate of return on the stock market minus the prevailing short-term interest rate. We use S&P 500 index returns from 1926 to 2005 from Center for Research in Security Press (CRSP) **month-end values**. Stock returns are the **continuously compounded** returns on the S&P 500 index, **including dividends**. Risk-free rate is the Treasury-bill rate. 
  - ✔️ `dfy` - Default Yield Spread is the difference between BAA and AAA-rated corporate bond yields. (`dfy = BAA - AAA`) 
  - ✔️ `dfr` - The Default Return Spread is the difference between long-term corporate bond and long-term government bond returns (`dfr = corpr - lty`). ❓ Check if `corpr` is the long-term corporate bond return.
  - ✔️ `infl` - Inflation is the Consumer Price Index. ❗ Because inflation information is released only in the following month, we **wait for one month before using it** in our monthly regressions
  - ✔️ `svar` - Stock Variance is computed as sum of squared daily returns on the S&P 500.
  - de (Dividend Payout Ratio)
  - ✔️ `lty` - Long-term government bond yield
  - ✔️ `tbl` - Treasury Bills. Treasury-bill rates from 1934 to 2005 are the 3-Month Treasury Bill: Secondary Market Rate from the economic research data base at the Federal Reserve Bank at St. Louis (FRED)
  - ✔️ `ltr` - Long Term Rate of Returns
  - ✔️ `tms` - The Term Spread is the difference between the long term yield on government bonds and the Treasury-bill. (`tms = lty - tbl`)
  - ✔️  `dp` - Dividend Price Ratio is the difference between the log of dividends and the log of prices. Dividends are 12-month moving sums of dividends paid on the S&P 500 index. (`D12` column). 
  - ✔️ `dy` - The Dividend Yield (d/y) is the difference between the log of dividends and the log of lagged prices. 
  - ✔️ `ep` - Earnings Price Ratio (e/p) is the difference between the log of earnings and the log of prices. Earnings are 12-month moving sums of earnings on the S&P 500 index. (`E12` column).
  - ✔️ `de` Dividend Payout Ratio (d/e) is the difference between the log of dividends and the log of earnings.
  - ✔️ `b/m` - Book-to-Market Ratio is the ratio of book value to market value for the Dow Jones Industrial Average.
  - ✔️ `ntis` - Net Equity Expansion is the ratio of 12-month moving sums of net issues by NYSE listed stocks divided by the total end-of-year market capitalization of NYSE stocks.
  - ✔️ `mr` one lag of the market return - once lagged `CRSP_SPvw`, or `CRSP_SPvwx` if we want to exclude dividends.
  

Data originally from Welch and Goyal (2008) and updated by Goyal (2022):

[A Comprehensive Look at the Empirical Performance of Equity Premium Prediction (with Ivo Welch), July 2008, Review of Financial Studies 21(4) 1455‒1508.](https://drive.google.com/file/d/1uvjBJ9D09T0_sp7kQppWpD-xelJ0KQhc/view)\
    [Original data (up to 2005) used in 2008 paper](https://drive.google.com/file/d/1T0pCslc2vxMDt7EFGI0MJ6mndeQvObBT/view?usp=sharing) \
    [Updated data (up to 2022)](https://docs.google.com/spreadsheets/d/1g4LOaRj4TvwJr9RIaA_nwrXXWTOy46bP/edit?usp=share_link&ouid=113571510202500088860&rtpof=true&sd=true)

All the links are provided on the [Goyal's website](https://sites.google.com/view/agoyal145).

## Notes from "Empirical Asset Pricing via ML"

I skipped the Intro and Ch1.

### CH 2: An Empirical Study of U.S. Equities:

- Mothly returns of approx 30,000 US stocks from March 1957 to Dec 2016. Data from CRSP.
- 8 marco variables as in Welch and Goyal 2008:
  - Dividend Price Ratio (dp)
  - Earnings Price Ratio (ep)
  - Book to Market Ratio (bm)
  - Net Equity Expansion (ntis)
  - Treasury bill rate (tbl)
  - Term Spread (tms)
  - Default Spread (dfs)
  - Stock variance (svar)
