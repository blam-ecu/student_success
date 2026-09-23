**Supporting Student Success Through an Interpretable Early Warning System**

This repository contains the Python implementation used in the study: 
  Lam, B., & Richardson, S. (2026). _Supporting Student Success Through an Interpretable Early Warning System in a First-Year Statistics Unit._

The workflow generates an educator-facing risk report that identifies students who may be at risk of failing a unit before the final examination period. Predictions are based on formative assessment results and enrolment characteristics, and are accompanied by interpretable risk classifications to support targeted intervention.

**Overview**

The workflow:

1. Imports student assessment and enrolment data.
2. Applies a trained machine learning ensemble model.
3. Calculates the probability of unit failure for each student.
4. Classifies students into risk categories.
5. Generates an Excel-based risk report for educators.
6. Provides model explainability using SHAP values.

**Required Data**

The workflow expects a dataset containing:

Student ID
Test 1 result
Test 2 result
Course code
Study mode
Attendance type
Fee category

Equivalent variables from other institutions may be substituted.
