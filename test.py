"""Smoke test rapide : python test.py"""
from scripts.run_preprocessing import main

if __name__ == "__main__":
    r = main()
    assert r["n_features_out"] == 18, r["n_features_out"]
    print("OK - prétraitement fonctionnel")
