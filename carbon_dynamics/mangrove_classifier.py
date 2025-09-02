"""
Classificateur de types de mangroves basé sur les caractéristiques hydrodynamiques.

Ce module classifie les mangroves en trois catégories principales :
- Marines : directement exposées aux marées océaniques
- Estuariennes : dans les zones de mélange eau douce/salée
- Terrestres : en bordure, moins influencées par les marées
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class MangroveTypeClassifier:
    """
    Classe pour classifier les types de mangroves selon leur exposition aux marées.
    """
    
    def __init__(self):
        """
        Initialise le classificateur de mangroves.
        """
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Types de mangroves
        self.mangrove_types = {
            0: 'marine',      # Directement exposées aux marées océaniques
            1: 'estuarine',   # Zones de mélange eau douce/salée
            2: 'terrestrial'  # Bordure, moins influencées par les marées
        }
    
    def extract_hydrodynamic_features(self, 
                                    inundation_frequency: np.ndarray,
                                    tidal_range: np.ndarray,
                                    distance_to_coast: np.ndarray,
                                    salinity_proxy: np.ndarray,
                                    elevation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extrait les caractéristiques hydrodynamiques pour la classification.
        
        Args:
            inundation_frequency: Fréquence d'inondation (0-1)
            tidal_range: Amplitude des marées (mètres)
            distance_to_coast: Distance à la côte (mètres)
            salinity_proxy: Proxy de salinité (indices spectraux)
            elevation: Élévation optionnelle (mètres)
            
        Returns:
            Array des features extraites
        """
        # Aplatir les arrays si nécessaire
        features_dict = {
            'inundation_freq': inundation_frequency.flatten() if inundation_frequency.ndim > 1 else inundation_frequency,
            'tidal_range': tidal_range.flatten() if tidal_range.ndim > 1 else tidal_range,
            'distance_coast': distance_to_coast.flatten() if distance_to_coast.ndim > 1 else distance_to_coast,
            'salinity_proxy': salinity_proxy.flatten() if salinity_proxy.ndim > 1 else salinity_proxy,
        }
        
        # Ajouter l'élévation si disponible
        if elevation is not None:
            features_dict['elevation'] = elevation.flatten() if elevation.ndim > 1 else elevation
        
        # S'assurer que tous les arrays ont la même longueur
        min_length = min(len(arr) for arr in features_dict.values())
        for key in features_dict:
            features_dict[key] = features_dict[key][:min_length]
        
        # Calculer des features dérivées avec la longueur corrigée
        derived_features = self._calculate_derived_features(features_dict)
        features_dict.update(derived_features)
        
        # Stocker les noms des features
        self.feature_names = list(features_dict.keys())
        
        # Convertir en array 2D pour sklearn
        feature_matrix = np.column_stack(list(features_dict.values()))
        
        return feature_matrix
    
    def _calculate_derived_features(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calcule des features dérivées pour améliorer la classification.
        
        Args:
            features_dict: Dictionnaire des features de base
            
        Returns:
            Dictionnaire des features dérivées
        """
        derived = {}
        
        # S'assurer que tous les arrays ont la même longueur
        min_length = min(len(arr) for arr in features_dict.values())
        
        # Extraire les features en s'assurant qu'elles ont la bonne longueur
        inundation_freq = features_dict['inundation_freq'][:min_length]
        distance_coast = features_dict['distance_coast'][:min_length]
        tidal_range = features_dict['tidal_range'][:min_length]
        salinity_proxy = features_dict['salinity_proxy'][:min_length]
        
        # Indice d'exposition marine (combinaison inundation et proximité côte)
        marine_exposure = (inundation_freq * 
                          (1 / (distance_coast + 1)))
        derived['marine_exposure'] = marine_exposure
        
        # Indice estuarien (influence mixte)
        estuarine_index = (tidal_range * 
                          inundation_freq * 
                          (1 - salinity_proxy))
        derived['estuarine_index'] = estuarine_index
        
        # Indice terrestre (faible inondation, éloignement)
        terrestrial_index = ((1 - inundation_freq) * 
                            distance_coast)
        derived['terrestrial_index'] = terrestrial_index
        
        # Gradient hydrique (changement d'inondation)
        if len(inundation_freq) > 1:
            hydric_gradient = np.gradient(inundation_freq)
        else:
            hydric_gradient = np.zeros_like(inundation_freq)
        derived['hydric_gradient'] = hydric_gradient
        
        # Stabilité hydrologique (inverse de la variabilité de marée)
        tidal_stability = 1 / (tidal_range + 0.1)
        derived['tidal_stability'] = tidal_stability
        
        return derived
    
    def create_training_labels(self, features: np.ndarray, 
                              method: str = 'rule_based') -> np.ndarray:
        """
        Crée des labels d'entraînement basés sur des règles expertes.
        
        Args:
            features: Matrice des features
            method: Méthode de labellisation ('rule_based', 'clustering')
            
        Returns:
            Array des labels (0: marine, 1: estuarine, 2: terrestrial)
        """
        if method == 'rule_based':
            return self._rule_based_labeling(features)
        elif method == 'clustering':
            return self._clustering_based_labeling(features)
        else:
            raise ValueError(f"Méthode non supportée: {method}")
    
    def _rule_based_labeling(self, features: np.ndarray) -> np.ndarray:
        """
        Labellisation basée sur des règles expertes.
        
        Args:
            features: Matrice des features
            
        Returns:
            Array des labels
        """
        labels = np.zeros(features.shape[0], dtype=int)
        
        # Extraire les indices des features importantes
        feature_idx = {name: i for i, name in enumerate(self.feature_names)}
        
        inundation_freq = features[:, feature_idx['inundation_freq']]
        distance_coast = features[:, feature_idx['distance_coast']]
        marine_exposure = features[:, feature_idx['marine_exposure']]
        terrestrial_index = features[:, feature_idx['terrestrial_index']]
        
        # Normaliser les distances pour les seuils
        distance_normalized = (distance_coast - distance_coast.min()) / (distance_coast.max() - distance_coast.min() + 1e-8)
        
        for i in range(len(labels)):
            # Mangroves marines : forte inondation, proximité côte
            if (inundation_freq[i] > 0.7 and 
                distance_normalized[i] < 0.3 and 
                marine_exposure[i] > np.percentile(marine_exposure, 70)):
                labels[i] = 0  # Marine
            
            # Mangroves terrestres : faible inondation, éloignement
            elif (inundation_freq[i] < 0.3 and 
                  distance_normalized[i] > 0.6 and 
                  terrestrial_index[i] > np.percentile(terrestrial_index, 60)):
                labels[i] = 2  # Terrestrial
            
            # Mangroves estuariennes : conditions intermédiaires
            else:
                labels[i] = 1  # Estuarine
        
        return labels
    
    def _clustering_based_labeling(self, features: np.ndarray) -> np.ndarray:
        """
        Labellisation basée sur clustering non supervisé.
        
        Args:
            features: Matrice des features
            
        Returns:
            Array des labels
        """
        from sklearn.cluster import KMeans
        
        # Normaliser les features pour le clustering
        features_scaled = self.scaler.fit_transform(features)
        
        # Clustering K-means avec 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Associer les clusters aux types de mangroves
        # Basé sur les centroïdes des clusters
        centroids = kmeans.cluster_centers_
        
        # Calculer des scores pour chaque cluster
        cluster_scores = {}
        feature_idx = {name: i for i, name in enumerate(self.feature_names)}
        
        for cluster_id in range(3):
            centroid = centroids[cluster_id]
            
            # Score marin : forte inondation + proximité côte
            marine_score = (centroid[feature_idx['marine_exposure']] + 
                           centroid[feature_idx['inundation_freq']])
            
            # Score terrestre : faible inondation + éloignement
            terrestrial_score = centroid[feature_idx['terrestrial_index']]
            
            cluster_scores[cluster_id] = {
                'marine': marine_score,
                'terrestrial': terrestrial_score
            }
        
        # Assigner les types basés sur les scores
        cluster_to_type = {}
        used_types = set()
        
        # Assigner le type marin au cluster avec le plus haut score marin
        marine_cluster = max(cluster_scores.keys(), 
                           key=lambda c: cluster_scores[c]['marine'])
        cluster_to_type[marine_cluster] = 0
        used_types.add(0)
        
        # Assigner le type terrestre au cluster avec le plus haut score terrestre
        remaining_clusters = [c for c in cluster_scores.keys() if c != marine_cluster]
        terrestrial_cluster = max(remaining_clusters, 
                                key=lambda c: cluster_scores[c]['terrestrial'])
        cluster_to_type[terrestrial_cluster] = 2
        used_types.add(2)
        
        # Le cluster restant devient estuarien
        estuarine_cluster = [c for c in cluster_scores.keys() 
                           if c not in cluster_to_type][0]
        cluster_to_type[estuarine_cluster] = 1
        
        # Convertir les labels de cluster en labels de type
        type_labels = np.array([cluster_to_type[cluster] for cluster in cluster_labels])
        
        return type_labels
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              test_size: float = 0.2) -> Dict:
        """
        Entraîne le classificateur.
        
        Args:
            features: Matrice des features
            labels: Labels de vérité terrain
            test_size: Proportion des données pour le test
            
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entraînement
        self.classifier.fit(X_train_scaled, y_train)
        
        # Prédictions
        y_pred_train = self.classifier.predict(X_train_scaled)
        y_pred_test = self.classifier.predict(X_test_scaled)
        
        # Métriques
        train_accuracy = np.mean(y_pred_train == y_train)
        test_accuracy = np.mean(y_pred_test == y_test)
        
        # Rapport de classification
        class_report = classification_report(
            y_test, y_pred_test, 
            target_names=list(self.mangrove_types.values()),
            output_dict=True
        )
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        
        # Importance des features
        feature_importance = dict(zip(
            self.feature_names, 
            self.classifier.feature_importances_
        ))
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance
        }
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédit les types de mangroves.
        
        Args:
            features: Matrice des features
            
        Returns:
            Tuple (prédictions, probabilités)
        """
        if not self.is_trained:
            raise ValueError("Le classificateur doit être entraîné avant la prédiction")
        
        features_scaled = self.scaler.transform(features)
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def classify_mangrove_map(self, 
                            inundation_frequency: np.ndarray,
                            tidal_range: np.ndarray,
                            distance_to_coast: np.ndarray,
                            salinity_proxy: np.ndarray,
                            elevation: Optional[np.ndarray] = None) -> Dict:
        """
        Classifie une carte complète de mangroves.
        
        Args:
            inundation_frequency: Carte de fréquence d'inondation
            tidal_range: Carte d'amplitude de marée
            distance_to_coast: Carte de distance à la côte
            salinity_proxy: Carte de proxy de salinité
            elevation: Carte d'élévation optionnelle
            
        Returns:
            Dictionnaire avec la carte classifiée et les statistiques
        """
        original_shape = inundation_frequency.shape
        
        # Extraire les features
        features = self.extract_hydrodynamic_features(
            inundation_frequency, tidal_range, distance_to_coast, 
            salinity_proxy, elevation
        )
        
        # Si le modèle n'est pas entraîné, utiliser la labellisation basée sur des règles
        if not self.is_trained:
            logger.info("Modèle non entraîné, utilisation de règles expertes")
            labels = self.create_training_labels(features, method='rule_based')
            training_metrics = self.train(features, labels)
            logger.info(f"Entraînement terminé - Précision: {training_metrics['test_accuracy']:.3f}")
        
        # Prédire
        predictions, probabilities = self.predict(features)
        
        # Reformater en carte 2D
        if len(original_shape) == 2:
            prediction_map = predictions.reshape(original_shape)
            probability_maps = {
                self.mangrove_types[i]: probabilities[:, i].reshape(original_shape)
                for i in range(len(self.mangrove_types))
            }
        else:
            # Si les données sont 1D, créer une forme arbitraire
            sqrt_size = int(np.sqrt(len(predictions)))
            prediction_map = predictions[:sqrt_size**2].reshape(sqrt_size, sqrt_size)
            probability_maps = {
                self.mangrove_types[i]: probabilities[:sqrt_size**2, i].reshape(sqrt_size, sqrt_size)
                for i in range(len(self.mangrove_types))
            }
        
        # Calculer les statistiques
        unique, counts = np.unique(predictions, return_counts=True)
        type_statistics = {}
        for type_id, count in zip(unique, counts):
            type_name = self.mangrove_types[type_id]
            percentage = (count / len(predictions)) * 100
            type_statistics[type_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        return {
            'prediction_map': prediction_map,
            'probability_maps': probability_maps,
            'type_statistics': type_statistics,
            'type_mapping': self.mangrove_types
        }
    
    def export_classification_results(self, results: Dict, output_path: str):
        """
        Exporte les résultats de classification.
        
        Args:
            results: Résultats de la classification
            output_path: Chemin de base pour les fichiers de sortie
        """
        import json
        import os
        
        # Exporter les statistiques en JSON
        stats_path = f"{output_path}_classification_stats.json"
        export_data = {
            'type_statistics': results['type_statistics'],
            'type_mapping': results['type_mapping']
        }
        
        with open(stats_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Exporter les cartes en format numpy
        maps_path = f"{output_path}_classification_maps.npz"
        np.savez_compressed(
            maps_path,
            prediction_map=results['prediction_map'],
            **results['probability_maps']
        )
        
        logger.info(f"Résultats de classification exportés:")
        logger.info(f"  - Statistiques: {stats_path}")
        logger.info(f"  - Cartes: {maps_path}")
        
        # Afficher un résumé
        logger.info("Répartition des types de mangroves:")
        for type_name, stats in results['type_statistics'].items():
            logger.info(f"  - {type_name}: {stats['percentage']:.1f}% ({stats['count']} pixels)")
