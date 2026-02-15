import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# ... (your imports) ...

# --- PAGE CONFIGURATION (Link Title & Icon) ---
# This sets the title for WhatsApp/Browser Tabs
st.set_page_config(
    page_title="Analytic Hierarchy Process-by Ahmad Fauzi",  # <--- This is the Title people will see
    page_icon="⚓",                          # <--- This is the little icon
    layout="wide"
)

# --- LOGIN SYSTEM ---
password = st.sidebar.text_input("Enter Password to Access:", type="password")

if password != "1234":  # Change "1234" to your real password
    st.sidebar.error("🔒 Access Denied")
    st.warning("Please enter the correct password in the sidebar to view the app.")
    st.stop()  # <--- THIS IS THE MAGIC COMMAND. It stops the app here!
    # --- YOUR MAIN APP CODE STARTS HERE ---
    st.sidebar.success("Logged in successfully")
    
    # ... (The rest of your existing code goes here) ...
    
    # ... (Paste all your Tab1, Tab2, Tab3 code here, indented) ...

# ---------------------------------------------------------
# 1. AHP ENGINE (Math Logic)
# ---------------------------------------------------------
class AHP:
    def __init__(self, criteria, comparison_matrix):
        self.criteria = criteria
        self.matrix = np.array(comparison_matrix, dtype=float)
        self.n = len(criteria)
        self.RI = {
            1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
            11: 1.51, 12: 1.54, 13: 1.56, 14: 1.57, 15: 1.59, 16: 1.60, 17: 1.61, 18: 1.62, 19: 1.63, 20: 1.63
        }
        self.weights = None

    def calculate_weights(self):
        column_sums = self.matrix.sum(axis=0)
        normalized_matrix = self.matrix / column_sums
        self.weights = normalized_matrix.mean(axis=1)
        return self.weights

    def check_consistency(self):
        if self.weights is None: self.calculate_weights()
        weighted_sum_vector = self.matrix.dot(self.weights)
        lambda_vector = weighted_sum_vector / self.weights
        lambda_max = lambda_vector.mean()
        ci = (lambda_max - self.n) / (self.n - 1)
        ri = self.RI.get(self.n, 1.63)
        cr = ci / ri if ri != 0 else 0
        return cr

    def suggest_improvement(self):
        if self.weights is None: self.calculate_weights()
        max_diff = 0
        suggestion = None
        for i in range(self.n):
            for j in range(i + 1, self.n):
                current_val = self.matrix[i, j]
                ratio = self.weights[i] / self.weights[j]
                
                # Clamp to 1/9 ... 9
                if ratio > 9: ideal_val = 9.0
                elif ratio < (1.0/9.0): ideal_val = 1.0/9.0
                else: ideal_val = ratio
                
                diff = abs(current_val - ideal_val)
                if diff > max_diff:
                    max_diff = diff
                    suggestion = (i, j, current_val, ideal_val)
        return suggestion

# ---------------------------------------------------------
# 2. STREAMLIT INTERFACE
# ---------------------------------------------------------
st.set_page_config(page_title="AHP Calculator {Beta}", layout="wide")
 
st.title("AHP Priority Calculator")

# --- SIDEBAR ---
st.sidebar.header("1. Define Criteria")
default_criteria = "Price\nQuality\nDesign\nSupport"

# 1. The Input Form (Mobile Friendly)
with st.sidebar.form(key='criteria_form'):
    # Note: We use 'st.text_area' here, NOT 'st.sidebar.text_area' 
    # because it is already inside the sidebar form.
    criteria_text = st.text_area("Type criteria (one per line):", value=default_criteria, height=250)
    submit_button = st.form_submit_button(label='✅ Update Criteria')

# 2. Process the Input
criteria = [c.strip() for c in criteria_text.split('\n') if c.strip()]

# 3. Check for Errors
count = len(criteria)
if count < 2:
    st.sidebar.error(f"⚠️ Total: {count} (Need at least 2)")
    st.stop()
elif count > 20:
    st.sidebar.error(f"⚠️ Total: {count} (Max recommended is 20)")
else:
    st.sidebar.success(f"✅ Total Criteria: {count}")

n = len(criteria)
matrix = np.ones((n, n))

# --- TABS ---
tab1, tab2 = st.tabs(["📝 Comparisons & Results", "🔢 Full Matrix"])

# --- TAB 1: SLIDERS & RESULTS ---
with tab1:
    st.markdown("### 2. Pairwise Comparisons")
    st.caption(f"Total comparisons: {int((n*(n-1))/2)}")
    st.divider()
    
    container = st.container()
    
    for i in range(n):
        for j in range(i + 1, n):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.markdown(f"<div style='text-align: right; font-weight: bold; padding-top: 10px;'>{criteria[i]}</div>", unsafe_allow_html=True)
            with col2:
                val = st.slider(f"{criteria[i]} vs {criteria[j]}", -9, 9, 0, 1, key=f"s_{i}_{j}", label_visibility="collapsed")
            with col3:
                st.markdown(f"<div style='text-align: left; font-weight: bold; padding-top: 10px;'>{criteria[j]}</div>", unsafe_allow_html=True)

            if val == 0: 
                actual_val = 1; desc = "Equal Importance"
            elif val > 0: 
                actual_val = val; desc = f"Favoring {criteria[i]} by {val}"
            else: 
                actual_val = 1 / abs(val); desc = f"Favoring {criteria[j]} by {abs(val)}"

            st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.8em; margin-bottom: 5px;'>{desc}</div>", unsafe_allow_html=True)
            st.markdown("---") 
            matrix[i, j] = actual_val
            matrix[j, i] = 1 / actual_val
    
    st.markdown("### 3. Analysis Results")
    
    if st.button("Calculate Results", type="primary", use_container_width=True):
        ahp = AHP(criteria, matrix)
        weights = ahp.calculate_weights()
        cr = ahp.check_consistency()

        # Create DataFrame
        results_df = pd.DataFrame({
            'Criteria': criteria,
            'Weight': weights,
        }).sort_values(by='Weight', ascending=False)

        # --- FIX: Start Index at 1 instead of 0 ---
        results_df.index = range(1, len(results_df) + 1)
        
        # Chart
        st.subheader("Visualization")
        chart = alt.Chart(results_df.reset_index()).mark_bar().encode(
            x=alt.X('Criteria', sort=None),
            y=alt.Y('Weight', scale=alt.Scale(domain=[0, 1])),
            tooltip=['Criteria', alt.Tooltip('Weight', format='.4f')]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Priorities Table")
            st.dataframe(results_df.style.format({'Weight': '{:.4f}'}), use_container_width=True)
            st.download_button("📥 Download Result (CSV)", results_df.to_csv().encode('utf-8'), "priorities.csv", "text/csv")

        with col2:
            st.subheader("Consistency")
            if cr < 0.1:
                st.success(f"CR: {cr:.1%}")
                st.caption("✅ Consistent")
            else:
                st.error(f"CR: {cr:.1%}")
                st.caption("⚠️ Inconsistent")
                sug = ahp.suggest_improvement()
                if sug:
                    i, j, curr, ideal = sug
                    crit_a, crit_b = criteria[i], criteria[j]
                    if ideal >= 1: dir_text = f"Favor **{crit_a}**"; val_text = f"{ideal:.2f}"
                    else: dir_text = f"Favor **{crit_b}**"; val_text = f"{1/ideal:.2f}"
                    st.warning("**Fix Suggestion:**")
                    st.markdown(f"Conflict: **{crit_a}** vs **{crit_b}**")
                    st.info(f"Try setting slider to: **{val_text}** ({dir_text})")

# --- TAB 2: MATRIX ---
with tab2:
    st.markdown("### Pairwise Comparison Matrix")
    matrix_df = pd.DataFrame(matrix, index=criteria, columns=criteria)
    st.dataframe(matrix_df)

    st.download_button("📥 Download Matrix (CSV)", matrix_df.to_csv().encode('utf-8'), "matrix.csv", "text/csv")



