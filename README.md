# RSRS
Reproduction of Everbright Securities' Market Time Based on Resistance Support Relative Strength

Here is a **GitHub README** based on the **RSRS (Resistance Support Relative Strength) Market Timing Model** from the uploaded report.

---

# 📌 **RSRS Market Timing Strategy: Resistance-Support Relative Strength (RSRS) Indicator**

This repository provides a full implementation of the **Resistance-Support Relative Strength (RSRS) model**, a **quantitative market timing strategy** based on technical analysis of **support and resistance levels**. The RSRS indicator dynamically assesses **relative strength shifts** between support and resistance levels, offering a **leading indicator** for market trend predictions.

## 🔹 **Project Overview**  
Traditional **moving average (MA) and MACD indicators** often exhibit significant lag, reducing their predictive effectiveness in volatile markets. The **RSRS indicator** innovates by considering the **relative strength of support vs. resistance** as a **dynamic variable**, rather than static levels. This allows for **more responsive and predictive market timing signals**.

### **Why RSRS?**
✔️ **Captures real-time market trend shifts** by evaluating resistance vs. support strength  
✔️ **Reduces lag compared to traditional technical indicators** (e.g., MA, MACD)  
✔️ **Effective in both bull and bear markets**, predicting trend reversals early  
✔️ **High correlation (75%) with future 2-week returns**, making it a reliable timing tool  

---

## 🚀 **Workflow**  

1️⃣ **Data Collection & Preprocessing**  
- Extracts **highs and lows** of stock indices (e.g., CSI 300, SSE 50, CSI 500)  
- Computes **rolling regression slope (β)** to quantify **support-resistance strength**  

2️⃣ **RSRS Indicator Calculation**  
- Uses **linear regression on N-day high/low prices**  
- The slope (β) represents **relative strength**:  
  ✔ **β > threshold** → Bullish signal (buy)  
  ✔ **β < threshold** → Bearish signal (sell)  

3️⃣ **Standardization & Bias Correction**  
- Converts β into **Z-score (standard deviation-based scaling)**  
- Implements **right-skewed standardization** for improved predictive power  

4️⃣ **Market Timing Strategy Implementation**  
- **Long-Only Strategy**: Buy when Z-score > S1, sell when Z-score < S2  
- **Long-Short Strategy**: Take short positions when bearish signals emerge  

5️⃣ **Performance Optimization**  
- **Price & volume filters** to **reduce false signals** and enhance profitability  
- **Multi-market validation** across different indices for robustness  

---

## 📈 **Results & Performance**  

✔️ **RSRS right-skewed Z-score strategy** applied to **CSI 300 (2005-2017)**:  
   - **Total return**: **1,573.60%**  
   - **Annualized return**: **25.82%**  
   - **Sharpe ratio**: **1.20**  
   - **Max drawdown**: **50.49%**  

✔️ **RSRS strategy on other indices**:  
   - **SSE 50 Index**: **1,432.36% return, 24.84% annualized**  
   - **CSI 500 Index**: **2,898.93% return, 32.39% annualized**  

🏆 **Key Features**  
✔ **High-frequency, dynamic market timing**  
✔ **Combines support-resistance analysis with statistical modeling**  
✔ **Outperforms traditional technical indicators (MA, MACD, Bollinger Bands)**  
✔ **Python-based implementation using Pandas, NumPy, Scikit-learn**  

---

## 📬 **Contact & Citation**  
If you find this project useful, feel free to reach out via **tenghanz@usc.edu** or contribute to the repository! 🚀  

---
