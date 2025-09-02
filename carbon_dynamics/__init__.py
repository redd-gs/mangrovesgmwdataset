"""
Module d'analyse des dynamiques de carbone dans les mangroves en fonction des marées.

Ce module permet d'analyser comment les différents types de mangroves (marines, estuariennes, terrestres)
séquestrent le carbone en fonction des cycles de marée, en utilisant des séries temporelles d'images satellites.
"""

from .hydrodynamic_model import HydrodynamicModel
from .carbon_sequestration import CarbonSequestrationAnalyzer
from .tidal_analysis import TidalAnalyzer
from .mangrove_classifier import MangroveTypeClassifier
from .time_series_processor import TimeSeriesProcessor

__all__ = [
    'HydrodynamicModel',
    'CarbonSequestrationAnalyzer', 
    'TidalAnalyzer',
    'MangroveTypeClassifier',
    'TimeSeriesProcessor'
]
