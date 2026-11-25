"""Non-régression : le module non supervisé doit rester importable.

Le refactoring de ``src/preprocessing`` a renommé ``ALL_FEATURES`` et
``load_raw_data`` ; une référence oubliée faisait échouer l'import de tout le
module. Ce test garde la trace de cette panne sans dépendre du jeu complet.
"""
import importlib


def test_import_unsupervised():
    module = importlib.import_module("src.unsupervised")

    assert hasattr(module, "UNSUPERVISED_MODELS")
    assert callable(module.load_raw)
    assert module.DEFAULT_DATA_PATH.name == "Employee_Attrition_DataSet.csv"
