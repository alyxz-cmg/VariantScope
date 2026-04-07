"""Streamlit demo for the Variant Impact Predictor."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from pipeline import run_pipeline
from feature_engineering import extract_features_from_string, parse_variant
from evaluation import get_roc_curves, evaluate_model, compute_shap_values

st.set_page_config(page_title="Variant Impact Predictor", layout="wide")
st.title("Variant Impact Predictor for Missense Mutations")