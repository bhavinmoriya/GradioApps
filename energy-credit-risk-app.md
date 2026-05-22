This code is a **credit risk scoring system** for an energy portfolio. Here's a breakdown:

---

## **1. Setup & Constants**
```python
REQUIRED_COLUMNS = {"counterparty", "segment", "country", ...}
```
- Defines **14 required columns** for input data (counterparty details, financials, risk metrics)

```python
RATING_SCORE = {"AAA": 0.02, "AA": 0.04, ..., "D": 1.0}
```
- Maps **credit ratings** to numeric risk scores (higher = worse)

---

---

## **2. Data Loading**
### `read_portfolio(path)`
- Reads a CSV into a **Polars DataFrame** (fast alternative to pandas)
- Validates all `REQUIRED_COLUMNS` exist, raises error if missing

---

---

## **3. Core Scoring Logic**
### `score_portfolio(frame)`

**Step 1: Normalize raw metrics to 0-1 scale**
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `utilization` | `exposure/limit` (clipped to 0-1.5) | How much of credit limit is used |
| `leverage_risk` | `leverage_ratio/7` | High leverage = higher risk |
| `liquidity_risk` | `1 - (liquidity_ratio/1.8)` | Low liquidity = higher risk |
| `hedge_gap` | `1 - (hedge_coverage_pct/100)` | Low hedge coverage = higher risk |
| `commodity_risk` | `commodity_sensitivity` | Sensitivity to commodity prices |
| `tenor_risk` | `contract_tenor_months/84` | Longer contracts = higher risk |
| `collateral_gap` | `1 - (collateral_pct/100)` | Low collateral = higher risk |
| `arrears_risk` | `days_past_due/60` | Payment delays |
| `esg_risk` | `1 if esg_incident else 0` | ESG incident flag |

**Step 2: Weighted Risk Score**
```
risk_score = (
    0.18 * rating_risk +
    0.14 * utilization +
    0.12 * leverage_risk +
    0.10 * liquidity_risk +
    0.10 * hedge_gap +
    0.11 * commodity_risk +
    0.08 * tenor_risk +
    0.08 * collateral_gap +
    0.06 * arrears_risk +
    0.03 * esg_risk
)
```
- **Credit rating** (18%) and **limit utilization** (14%) have highest weights
- **ESG incidents** (3%) have lowest weight

**Step 3: Risk Classification**
| Score Range | Tier |
|-------------|------|
| ≥ 0.65 | Critical |
| ≥ 0.48 | High |
| ≥ 0.32 | Watch |
| < 0.32 | Acceptable |

**Step 4: Risk-Weighted Exposure**
- `risk_weighted_exposure = risk_score * exposure_musd`
- Prioritizes high-risk, high-exposure counterparties

---

---

## **4. Summary & Reporting**

### `portfolio_summary(scored)`
- **Groups by risk tier** and calculates:
  - Number of counterparties
  - Total exposure
  - Total risk-weighted exposure
  - Average risk score

### `analyst_memo(scored)`
- Generates a **text summary** with:
  - Total portfolio exposure
  - Total risk-weighted exposure
  - Count of High/Critical names
  - **Top 3 riskiest counterparties** with their risk drivers

### `_risk_drivers(row)`
- Identifies **specific risk factors** for each counterparty:
  - High limit utilization (>85%)
  - Elevated leverage (>5x)
  - Thin liquidity (<1.0)
  - Low hedge coverage (<35%)
  - High commodity sensitivity (>75%)
  - Payment arrears (>15 days)
  - ESG incident

---
---
**In short**: This system **scores counterparty risk** using weighted financial/operational metrics, classifies them into tiers, and generates actionable reports for analysts.
