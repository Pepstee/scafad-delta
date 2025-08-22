#!/usr/bin/env python3
"""
SCAFAD Layer 1: Research Metrics Collection Module
=================================================

Publication-grade metrics collection with statistical significance testing,
comparative analysis, and ablation studies for academic research.

Integrates with existing Enhanced Anomaly Preservation Guard.

Author: SCAFAD Research Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import json
import time
import logging
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque
import warnings
import copy
import pickle

# Scientific computing
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score
)
from sklearn.model_selection import cross_val_score, permutation_test_score
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.descriptivestats import describe
from statsmodels.stats.power import ttest_power
from statsmodels.tsa.stattools import adfuller
import pingouin as pg  # For advanced statistical tests


class ResearchMetricType(Enum):
    """Types of research metrics"""
    PRESERVATION_EFFECTIVENESS = "preservation_effectiveness"
    LATENCY_PERFORMANCE = "latency_performance"
    MEMORY_EFFICIENCY = "memory_efficiency"
    PRIVACY_UTILITY_TRADEOFF = "privacy_utility_tradeoff"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    COMPARATIVE_BASELINE = "comparative_baseline"
    ABLATION_STUDY = "ablation_study"
    SCALABILITY_ANALYSIS = "scalability_analysis"
    ROBUSTNESS_TESTING = "robustness_testing"


class ExperimentalCondition(Enum):
    """Experimental conditions for controlled testing"""
    BASELINE = "baseline"
    PROPOSED_METHOD = "proposed_method"
    ABLATED_COMPONENT = "ablated_component"
    COMPETITIVE_BASELINE = "competitive_baseline"
    STRESS_TEST = "stress_test"


@dataclass
class StatisticalTest:
    """Statistical significance test result"""
    test_name: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    
    # Test metadata
    alpha_level: float = 0.05
    power: float = 0.0
    sample_size: int = 0
    assumptions_met: bool = True
    assumption_violations: List[str] = field(default_factory=list)
    
    # Interpretation
    is_significant: bool = field(init=False)
    effect_size_interpretation: str = field(init=False)
    
    def __post_init__(self):
        self.is_significant = self.p_value < self.alpha_level
        self.effect_size_interpretation = self._interpret_effect_size()
    
    def _interpret_effect_size(self) -> str:
        """Interpret effect size magnitude"""
        abs_effect = abs(self.effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"


@dataclass
class ExperimentalResult:
    """Single experimental result with metadata"""
    experiment_id: str
    condition: ExperimentalCondition
    metric_values: Dict[str, float]
    
    # Experimental metadata
    timestamp: float
    sample_size: int
    execution_time_ms: float
    memory_usage_mb: float
    
    # Context
    data_characteristics: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, str] = field(default_factory=dict)
    
    # Quality indicators
    data_quality_score: float = 1.0
    measurement_confidence: float = 1.0
    outlier_score: float = 0.0


@dataclass
class ComparativeAnalysis:
    """Comparative analysis between methods"""
    baseline_method: str
    proposed_method: str
    comparison_metrics: Dict[str, Dict[str, float]]
    
    # Statistical comparison
    statistical_tests: Dict[str, StatisticalTest]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Summary statistics
    improvement_ratios: Dict[str, float]
    significance_summary: str
    overall_winner: str
    
    # Detailed analysis
    metric_rankings: Dict[str, List[str]]
    trade_off_analysis: Dict[str, Any]


@dataclass
class AblationStudyResult:
    """Ablation study result showing component importance"""
    component_name: str
    baseline_performance: Dict[str, float]
    ablated_performance: Dict[str, float]
    
    # Impact analysis
    performance_drop: Dict[str, float]
    relative_importance: float
    critical_metrics: List[str]
    
    # Statistical validation
    significance_tests: Dict[str, StatisticalTest]
    confidence_level: float


class ResearchMetricsCollector:
    """Publication-grade metrics collection and analysis"""
    
    def __init__(self, preservation_guard, config: Dict[str, Any] = None):
        self.preservation_guard = preservation_guard
        self.config = config or {}
        
        # Metrics storage
        self.experimental_results = deque(maxlen=self.config.get('max_results', 10000))
        self.baseline_results = {}
        self.ablation_results = {}
        
        # Statistical configuration
        self.alpha_level = self.config.get('alpha_level', 0.05)
        self.power_threshold = self.config.get('power_threshold', 0.8)
        self.effect_size_threshold = self.config.get('effect_size_threshold', 0.2)
        
        # Experimental design
        self.controlled_variables = self.config.get('controlled_variables', [])
        self.random_seed = self.config.get('random_seed', 42)
        np.random.seed(self.random_seed)
        
        # Publication metrics
        self.publication_metrics = PublicationMetricsGenerator(self)
        
        self.logger = logging.getLogger(__name__ + ".ResearchMetricsCollector")
    
    async def collect_publication_metrics(self, experiment_name: str,
                                        conditions: List[ExperimentalCondition],
                                        test_scenarios: List[Dict[str, Any]],
                                        num_repetitions: int = 30) -> Dict[str, Any]:
        """Collect comprehensive metrics suitable for academic publication"""
        
        start_time = time.perf_counter()
        
        publication_data = {
            'experiment_metadata': {
                'name': experiment_name,
                'timestamp': time.time(),
                'conditions': [c.value for c in conditions],
                'scenarios': len(test_scenarios),
                'repetitions': num_repetitions,
                'total_experiments': len(conditions) * len(test_scenarios) * num_repetitions
            },
            'experimental_results': {},
            'statistical_analysis': {},
            'comparative_analysis': {},
            'effect_size_analysis': {},
            'publication_summary': {}
        }
        
        # Run experiments for each condition
        for condition in conditions:
            self.logger.info(f"Running experiments for condition: {condition.value}")
            
            condition_results = await self._run_condition_experiments(
                condition, test_scenarios, num_repetitions
            )
            
            publication_data['experimental_results'][condition.value] = condition_results
        
        # Perform statistical analysis
        self.logger.info("Performing statistical analysis")
        statistical_analysis = await self._perform_comprehensive_statistical_analysis(
            publication_data['experimental_results']
        )
        publication_data['statistical_analysis'] = statistical_analysis
        
        # Comparative analysis between conditions
        if len(conditions) > 1:
            self.logger.info("Performing comparative analysis")
            comparative_analysis = await self._perform_comparative_analysis(
                publication_data['experimental_results']
            )
            publication_data['comparative_analysis'] = comparative_analysis
        
        # Effect size analysis
        self.logger.info("Calculating effect sizes")
        effect_size_analysis = await self._calculate_effect_sizes(
            publication_data['experimental_results']
        )
        publication_data['effect_size_analysis'] = effect_size_analysis
        
        # Generate publication summary
        publication_summary = await self._generate_publication_summary(
            publication_data
        )
        publication_data['publication_summary'] = publication_summary
        
        total_time = (time.perf_counter() - start_time) / 60  # Convert to minutes
        publication_data['experiment_metadata']['total_duration_minutes'] = total_time
        
        self.logger.info(f"Publication metrics collection completed in {total_time:.2f} minutes")
        
        return publication_data
    
    async def _run_condition_experiments(self, condition: ExperimentalCondition,
                                       test_scenarios: List[Dict[str, Any]],
                                       num_repetitions: int) -> Dict[str, Any]:
        """Run experiments for a specific condition"""
        
        condition_results = {
            'condition': condition.value,
            'scenario_results': {},
            'aggregated_metrics': {},
            'raw_measurements': defaultdict(list),
            'quality_indicators': {}
        }
        
        for scenario_idx, scenario in enumerate(test_scenarios):
            scenario_id = f"scenario_{scenario_idx}"
            scenario_results = []
            
            for rep in range(num_repetitions):
                try:
                    # Configure system for this condition
                    configured_system = await self._configure_system_for_condition(condition)
                    
                    # Run single experiment
                    result = await self._run_single_experiment(
                        configured_system, scenario, condition, f"{scenario_id}_rep_{rep}"
                    )
                    
                    scenario_results.append(result)
                    
                    # Collect raw measurements
                    for metric, value in result.metric_values.items():
                        condition_results['raw_measurements'][metric].append(value)
                    
                except Exception as e:
                    self.logger.error(f"Experiment failed for {scenario_id} rep {rep}: {e}")
            
            condition_results['scenario_results'][scenario_id] = scenario_results
        
        # Calculate aggregated metrics
        condition_results['aggregated_metrics'] = self._calculate_aggregated_metrics(
            condition_results['raw_measurements']
        )
        
        # Assess data quality
        condition_results['quality_indicators'] = self._assess_data_quality(
            condition_results['raw_measurements']
        )
        
        return condition_results
    
    async def _configure_system_for_condition(self, condition: ExperimentalCondition):
        """Configure preservation guard for experimental condition"""
        
        if condition == ExperimentalCondition.BASELINE:
            # Minimal configuration - basic preservation only
            config = {
                'enable_neural_encoder': False,
                'enable_adaptive_optimizer': False,
                'preservation_strategy': 'statistical_bounds'
            }
        
        elif condition == ExperimentalCondition.PROPOSED_METHOD:
            # Full system with all enhancements
            config = {
                'enable_neural_encoder': True,
                'enable_adaptive_optimizer': True,
                'preservation_strategy': 'comprehensive',
                'enable_semantic_preservation': True,
                'enable_privacy_utility_optimizer': True
            }
        
        elif condition == ExperimentalCondition.ABLATED_COMPONENT:
            # System with specific components removed
            config = {
                'enable_neural_encoder': False,  # Ablate neural component
                'enable_adaptive_optimizer': True,
                'preservation_strategy': 'information_theoretic'
            }
        
        elif condition == ExperimentalCondition.COMPETITIVE_BASELINE:
            # Configuration matching competitive approaches
            config = {
                'enable_neural_encoder': False,
                'enable_adaptive_optimizer': False,
                'preservation_strategy': 'feature_vectors'
            }
        
        else:  # STRESS_TEST
            # High-stress configuration
            config = {
                'enable_neural_encoder': True,
                'enable_adaptive_optimizer': True,
                'preservation_strategy': 'comprehensive',
                'stress_test_mode': True,
                'max_processing_latency_ms': 1  # Very tight constraint
            }
        
        # Create configured preservation guard
        configured_guard = copy.deepcopy(self.preservation_guard)
        # Apply configuration changes (simplified - real implementation would be more detailed)
        
        return configured_guard
    
    async def _run_single_experiment(self, configured_system, scenario: Dict[str, Any],
                                   condition: ExperimentalCondition, experiment_id: str) -> ExperimentalResult:
        """Run a single controlled experiment"""
        
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        # Create test data
        original_data = scenario.copy()
        processed_data = await self._apply_processing_transformations(original_data)
        
        # Measure preservation effectiveness
        assessment = await configured_system.assess_preservation_impact(
            original_data, processed_data, experiment_id
        )
        
        execution_time = (time.perf_counter() - start_time) * 1000
        memory_usage = self._get_memory_usage() - start_memory
        
        # Collect comprehensive metrics
        metric_values = {
            'preservation_effectiveness': assessment.preservation_effectiveness,
            'processing_latency_ms': execution_time,
            'memory_overhead_mb': memory_usage,
            'confidence_score': assessment.confidence_score,
            'information_loss': assessment.information_loss,
            'entropy_preserved': assessment.entropy_preserved,
            'critical_violations': len(assessment.critical_violations),
            'warning_violations': len(assessment.warning_violations)
        }
        
        # Add condition-specific metrics
        if hasattr(assessment, 'privacy_level'):
            metric_values['privacy_level'] = assessment.privacy_level
        
        if hasattr(assessment, 'utility_level'):
            metric_values['utility_level'] = assessment.utility_level
        
        # Data characteristics
        data_characteristics = {
            'field_count': len(original_data),
            'numeric_fields': sum(1 for v in original_data.values() if isinstance(v, (int, float))),
            'complex_fields': sum(1 for v in original_data.values() if isinstance(v, (dict, list))),
            'data_size_bytes': len(json.dumps(original_data)),
            'anomaly_indicators': len([k for k in original_data.keys() if 'anomaly' in k.lower()])
        }
        
        return ExperimentalResult(
            experiment_id=experiment_id,
            condition=condition,
            metric_values=metric_values,
            timestamp=time.time(),
            sample_size=1,
            execution_time_ms=execution_time,
            memory_usage_mb=memory_usage,
            data_characteristics=data_characteristics,
            configuration={'condition': condition.value},
            environment_info=self._get_environment_info()
        )
    
    async def _perform_comprehensive_statistical_analysis(self, experimental_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis of results"""
        
        analysis = {
            'normality_tests': {},
            'descriptive_statistics': {},
            'hypothesis_tests': {},
            'power_analysis': {},
            'assumptions_validation': {}
        }
        
        # Extract metrics for analysis
        all_metrics = set()
        for condition_data in experimental_results.values():
            all_metrics.update(condition_data['raw_measurements'].keys())
        
        for metric in all_metrics:
            metric_analysis = {}
            
            # Collect data for this metric across conditions
            metric_data = {}
            for condition, condition_data in experimental_results.items():
                if metric in condition_data['raw_measurements']:
                    metric_data[condition] = condition_data['raw_measurements'][metric]
            
            if len(metric_data) < 2:
                continue
            
            # Normality tests
            normality_results = {}
            for condition, data in metric_data.items():
                if len(data) >= 8:  # Minimum for Shapiro-Wilk
                    stat, p_val = stats.shapiro(data)
                    normality_results[condition] = {
                        'statistic': stat,
                        'p_value': p_val,
                        'is_normal': p_val > self.alpha_level
                    }
            
            metric_analysis['normality_tests'] = normality_results
            
            # Descriptive statistics
            descriptive_stats = {}
            for condition, data in metric_data.items():
                descriptive_stats[condition] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'median': np.median(data),
                    'iqr': np.percentile(data, 75) - np.percentile(data, 25),
                    'min': np.min(data),
                    'max': np.max(data),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'cv': np.std(data) / np.mean(data) if np.mean(data) != 0 else 0
                }
            
            metric_analysis['descriptive_statistics'] = descriptive_stats
            
            # Hypothesis tests (if we have exactly 2 conditions)
            conditions = list(metric_data.keys())
            if len(conditions) == 2:
                data1 = metric_data[conditions[0]]
                data2 = metric_data[conditions[1]]
                
                # Check normality assumption
                both_normal = (
                    normality_results.get(conditions[0], {}).get('is_normal', False) and
                    normality_results.get(conditions[1], {}).get('is_normal', False)
                )
                
                if both_normal:
                    # Use t-test
                    stat, p_val = stats.ttest_ind(data1, data2)
                    test_name = "Independent t-test"
                else:
                    # Use Mann-Whitney U test
                    stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    test_name = "Mann-Whitney U test"
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1) + 
                                    (len(data2) - 1) * np.var(data2)) / 
                                   (len(data1) + len(data2) - 2))
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                
                # Confidence interval for difference in means
                se_diff = pooled_std * np.sqrt(1/len(data1) + 1/len(data2))
                df = len(data1) + len(data2) - 2
                t_critical = stats.t.ppf(1 - self.alpha_level/2, df)
                mean_diff = np.mean(data1) - np.mean(data2)
                ci_lower = mean_diff - t_critical * se_diff
                ci_upper = mean_diff + t_critical * se_diff
                
                hypothesis_test = StatisticalTest(
                    test_name=test_name,
                    test_statistic=stat,
                    p_value=p_val,
                    effect_size=cohens_d,
                    confidence_interval=(ci_lower, ci_upper),
                    alpha_level=self.alpha_level,
                    sample_size=len(data1) + len(data2),
                    assumptions_met=both_normal
                )
                
                metric_analysis['hypothesis_test'] = asdict(hypothesis_test)
            
            analysis[metric] = metric_analysis
        
        return analysis
    
    async def _perform_comparative_analysis(self, experimental_results: Dict[str, Any]) -> Dict[str, ComparativeAnalysis]:
        """Perform pairwise comparative analysis between conditions"""
        
        comparative_analyses = {}
        conditions = list(experimental_results.keys())
        
        # Get all metrics
        all_metrics = set()
        for condition_data in experimental_results.values():
            all_metrics.update(condition_data['raw_measurements'].keys())
        
        # Compare each pair of conditions
        for i, condition1 in enumerate(conditions):
            for condition2 in conditions[i+1:]:
                comparison_key = f"{condition1}_vs_{condition2}"
                
                comparison_metrics = {}
                statistical_tests = {}
                effect_sizes = {}
                confidence_intervals = {}
                improvement_ratios = {}
                
                for metric in all_metrics:
                    data1 = experimental_results[condition1]['raw_measurements'].get(metric, [])
                    data2 = experimental_results[condition2]['raw_measurements'].get(metric, [])
                    
                    if not data1 or not data2:
                        continue
                    
                    # Basic comparison metrics
                    mean1, mean2 = np.mean(data1), np.mean(data2)
                    std1, std2 = np.std(data1), np.std(data2)
                    
                    comparison_metrics[metric] = {
                        condition1: {'mean': mean1, 'std': std1},
                        condition2: {'mean': mean2, 'std': std2}
                    }
                    
                    # Statistical test
                    try:
                        # Check for normality
                        _, p_norm1 = stats.shapiro(data1)
                        _, p_norm2 = stats.shapiro(data2)
                        both_normal = p_norm1 > 0.05 and p_norm2 > 0.05
                        
                        if both_normal:
                            stat, p_val = stats.ttest_ind(data1, data2)
                            test_name = "t-test"
                        else:
                            stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            test_name = "Mann-Whitney U"
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1) + 
                                            (len(data2) - 1) * np.var(data2)) / 
                                           (len(data1) + len(data2) - 2))
                        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                        
                        statistical_tests[metric] = StatisticalTest(
                            test_name=test_name,
                            test_statistic=stat,
                            p_value=p_val,
                            effect_size=cohens_d,
                            confidence_interval=(0, 0),  # Simplified
                            sample_size=len(data1) + len(data2),
                            assumptions_met=both_normal
                        )
                        
                        effect_sizes[metric] = cohens_d
                        
                        # Improvement ratio
                        if mean2 != 0:
                            improvement_ratios[metric] = (mean1 - mean2) / abs(mean2)
                        else:
                            improvement_ratios[metric] = 0
                        
                    except Exception as e:
                        self.logger.debug(f"Statistical test failed for {metric}: {e}")
                
                # Determine overall winner
                significant_improvements = sum(
                    1 for test in statistical_tests.values() 
                    if test.is_significant and test.effect_size > 0
                )
                
                significant_degradations = sum(
                    1 for test in statistical_tests.values() 
                    if test.is_significant and test.effect_size < 0
                )
                
                if significant_improvements > significant_degradations:
                    overall_winner = condition1
                elif significant_degradations > significant_improvements:
                    overall_winner = condition2
                else:
                    overall_winner = "tie"
                
                # Create comparative analysis
                comparative_analyses[comparison_key] = ComparativeAnalysis(
                    baseline_method=condition2,
                    proposed_method=condition1,
                    comparison_metrics=comparison_metrics,
                    statistical_tests=statistical_tests,
                    effect_sizes=effect_sizes,
                    confidence_intervals=confidence_intervals,
                    improvement_ratios=improvement_ratios,
                    significance_summary=f"{significant_improvements} improvements, {significant_degradations} degradations",
                    overall_winner=overall_winner,
                    metric_rankings={},  # Simplified
                    trade_off_analysis={}  # Simplified
                )
        
        return comparative_analyses
    
    async def _calculate_effect_sizes(self, experimental_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive effect size analysis"""
        
        effect_analysis = {
            'cohens_d': {},
            'eta_squared': {},
            'practical_significance': {},
            'magnitude_interpretation': {}
        }
        
        # Get baseline condition (assume first condition is baseline)
        conditions = list(experimental_results.keys())
        if len(conditions) < 2:
            return effect_analysis
        
        baseline_condition = conditions[0]
        baseline_data = experimental_results[baseline_condition]['raw_measurements']
        
        # Calculate effect sizes relative to baseline
        for condition in conditions[1:]:
            condition_data = experimental_results[condition]['raw_measurements']
            
            for metric in baseline_data.keys():
                if metric in condition_data:
                    baseline_values = baseline_data[metric]
                    condition_values = condition_data[metric]
                    
                    # Cohen's d
                    mean_diff = np.mean(condition_values) - np.mean(baseline_values)
                    pooled_std = np.sqrt((np.var(baseline_values) + np.var(condition_values)) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    effect_key = f"{condition}_vs_{baseline_condition}_{metric}"
                    effect_analysis['cohens_d'][effect_key] = cohens_d
                    
                    # Practical significance (domain-specific thresholds)
                    practical_threshold = self._get_practical_significance_threshold(metric)
                    is_practically_significant = abs(mean_diff) > practical_threshold
                    effect_analysis['practical_significance'][effect_key] = is_practically_significant
                    
                    # Magnitude interpretation
                    magnitude = self._interpret_effect_magnitude(cohens_d)
                    effect_analysis['magnitude_interpretation'][effect_key] = magnitude
        
        return effect_analysis
    
    def _get_practical_significance_threshold(self, metric: str) -> float:
        """Get practical significance threshold for metric"""
        
        thresholds = {
            'preservation_effectiveness': 0.02,  # 2% improvement is meaningful
            'processing_latency_ms': 1.0,       # 1ms improvement is meaningful
            'memory_overhead_mb': 5.0,          # 5MB reduction is meaningful
            'information_loss': 0.01             # 1% reduction is meaningful
        }
        
        return thresholds.get(metric, 0.1)  # Default 10% threshold
    
    def _interpret_effect_magnitude(self, effect_size: float) -> str:
        """Interpret effect size magnitude"""
        
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    async def _generate_publication_summary(self, publication_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate publication-ready summary"""
        
        summary = {
            'key_findings': [],
            'statistical_significance': {},
            'effect_sizes': {},
            'performance_summary': {},
            'limitations': [],
            'recommendations': []
        }
        
        # Extract key findings
        if 'comparative_analysis' in publication_data:
            for comparison, analysis in publication_data['comparative_analysis'].items():
                if analysis.overall_winner != "tie":
                    summary['key_findings'].append(
                        f"{analysis.overall_winner} significantly outperforms {analysis.baseline_method} "
                        f"({analysis.significance_summary})"
                    )
        
        # Statistical significance summary
        significant_results = 0
        total_tests = 0
        
        if 'statistical_analysis' in publication_data:
            for metric, analysis in publication_data['statistical_analysis'].items():
                if 'hypothesis_test' in analysis:
                    total_tests += 1
                    if analysis['hypothesis_test']['is_significant']:
                        significant_results += 1
        
        summary['statistical_significance'] = {
            'significant_results': significant_results,
            'total_tests': total_tests,
            'significance_rate': significant_results / total_tests if total_tests > 0 else 0
        }
        
        # Performance summary
        conditions = list(publication_data['experimental_results'].keys())
        if conditions:
            best_condition = max(
                conditions,
                key=lambda c: publication_data['experimental_results'][c]['aggregated_metrics'].get(
                    'preservation_effectiveness', {}).get('mean', 0)
            )
            
            summary['performance_summary'] = {
                'best_overall_method': best_condition,
                'key_metrics': publication_data['experimental_results'][best_condition]['aggregated_metrics']
            }
        
        # Limitations
        summary['limitations'] = [
            "Results based on controlled experimental scenarios",
            "Limited to specific data types and processing configurations",
            "Evaluation metrics may not capture all real-world performance aspects"
        ]
        
        # Recommendations for future work
        summary['recommendations'] = [
            "Validate results on larger-scale production deployments",
            "Investigate performance under adversarial conditions",
            "Extend evaluation to additional data types and domains"
        ]
        
        return summary
    
    # Helper methods
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get environment information"""
        import platform
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0]
        }
    
    def _calculate_aggregated_metrics(self, raw_measurements: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate aggregated statistics for metrics"""
        
        aggregated = {}
        
        for metric, values in raw_measurements.items():
            if values:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p25': np.percentile(values, 25),
                    'p75': np.percentile(values, 75),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                    'count': len(values)
                }
        
        return aggregated
    
    def _assess_data_quality(self, raw_measurements: Dict[str, List[float]]) -> Dict[str, Any]:
        """Assess quality of experimental data"""
        
        quality_indicators = {
            'outlier_analysis': {},
            'completeness': {},
            'consistency': {},
            'overall_quality_score': 0.0
        }
        
        quality_scores = []
        
        for metric, values in raw_measurements.items():
            if not values:
                continue
            
            # Outlier detection using IQR method
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            
            outlier_ratio = len(outliers) / len(values)
            quality_indicators['outlier_analysis'][metric] = {
                'outlier_count': len(outliers),
                'outlier_ratio': outlier_ratio,
                'outlier_quality_score': max(0, 1 - outlier_ratio * 2)  # Penalty for outliers
            }
            
            # Completeness (no missing values in this simplified case)
            completeness_score = 1.0
            quality_indicators['completeness'][metric] = completeness_score
            
            # Consistency (low coefficient of variation is good)
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            consistency_score = max(0, 1 - cv)
            quality_indicators['consistency'][metric] = consistency_score
            
            # Overall metric quality
            metric_quality = np.mean([
                quality_indicators['outlier_analysis'][metric]['outlier_quality_score'],
                completeness_score,
                consistency_score
            ])
            quality_scores.append(metric_quality)
        
        quality_indicators['overall_quality_score'] = np.mean(quality_scores) if quality_scores else 0.0
        
        return quality_indicators
    
    async def _apply_processing_transformations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply standard processing transformations for testing"""
        
        processed = copy.deepcopy(data)
        
        # Simulate various processing stages
        # Remove debug fields
        processed.pop('debug_info', None)
        processed.pop('internal_state', None)
        
        # Hash sensitive fields
        if 'user_id' in processed:
            processed['user_id'] = f"hash_{hash(processed['user_id'])}"
        
        # Add noise to numeric fields (simulate privacy protection)
        for key, value in list(processed.items()):
            if isinstance(value, (int, float)) and 'time' in key.lower():
                noise = np.random.normal(0, abs(value) * 0.01)  # 1% noise
                processed[key] = value + noise
        
        return processed


class PublicationMetricsGenerator:
    """Generate publication-ready metrics, tables, and figures"""
    
    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__ + ".PublicationMetricsGenerator")
    
    def generate_performance_comparison_table(self, publication_data: Dict[str, Any]) -> pd.DataFrame:
        """Generate LaTeX-ready performance comparison table"""
        
        experimental_results = publication_data['experimental_results']
        
        # Extract metrics for table
        table_data = []
        
        for condition, results in experimental_results.items():
            aggregated = results['aggregated_metrics']
            
            row = {
                'Method': condition.replace('_', ' ').title(),
                'Preservation Effectiveness': f"{aggregated.get('preservation_effectiveness', {}).get('mean', 0):.3f} ± {aggregated.get('preservation_effectiveness', {}).get('std', 0):.3f}",
                'Processing Latency (ms)': f"{aggregated.get('processing_latency_ms', {}).get('mean', 0):.2f} ± {aggregated.get('processing_latency_ms', {}).get('std', 0):.2f}",
                'Memory Overhead (MB)': f"{aggregated.get('memory_overhead_mb', {}).get('mean', 0):.1f} ± {aggregated.get('memory_overhead_mb', {}).get('std', 0):.1f}",
                'Information Loss': f"{aggregated.get('information_loss', {}).get('mean', 0):.3f} ± {aggregated.get('information_loss', {}).get('std', 0):.3f}",
                'Confidence Score': f"{aggregated.get('confidence_score', {}).get('mean', 0):.3f} ± {aggregated.get('confidence_score', {}).get('std', 0):.3f}"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def generate_statistical_significance_table(self, publication_data: Dict[str, Any]) -> pd.DataFrame:
        """Generate statistical significance summary table"""
        
        table_data = []
        
        if 'comparative_analysis' in publication_data:
            for comparison, analysis in publication_data['comparative_analysis'].items():
                for metric, test in analysis.statistical_tests.items():
                    row = {
                        'Comparison': comparison.replace('_', ' vs ').title(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Test': test.test_name,
                        'Test Statistic': f"{test.test_statistic:.4f}",
                        'p-value': f"{test.p_value:.4f}",
                        'Effect Size (Cohen\'s d)': f"{test.effect_size:.3f}",
                        'Significant': "Yes" if test.is_significant else "No",
                        'Effect Magnitude': test.effect_size_interpretation.title()
                    }
                    table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def generate_roc_curves(self, experimental_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ROC curves for anomaly detection performance"""
        
        roc_data = {}
        
        # This would typically require ground truth labels
        # For demonstration, we'll simulate based on preservation effectiveness
        
        for condition, results in experimental_results.items():
            preservation_scores = results['raw_measurements'].get('preservation_effectiveness', [])
            
            if not preservation_scores:
                continue
            
            # Simulate binary classification (high preservation = positive)
            threshold = np.median(preservation_scores)
            y_true = [1 if score > threshold else 0 for score in preservation_scores]
            y_scores = preservation_scores
            
            try:
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                
                roc_data[condition] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'auc': roc_auc
                }
            except Exception as e:
                self.logger.debug(f"ROC curve generation failed for {condition}: {e}")
        
        return roc_data
    
    def generate_latex_table(self, df: pd.DataFrame, caption: str, label: str) -> str:
        """Generate LaTeX table code"""
        
        latex_table = df.to_latex(
            index=False,
            escape=False,
            column_format='l' + 'c' * (len(df.columns) - 1),
            caption=caption,
            label=label,
            position='htbp'
        )
        
        return latex_table
    
    def generate_ablation_study_results(self, ablation_data: Dict[str, AblationStudyResult]) -> Dict[str, Any]:
        """Generate comprehensive ablation study analysis"""
        
        ablation_summary = {
            'component_importance_ranking': [],
            'performance_impact_analysis': {},
            'critical_components': [],
            'statistical_validation': {}
        }
        
        # Rank components by importance
        component_scores = []
        for component, result in ablation_data.items():
            avg_impact = np.mean(list(result.performance_drop.values()))
            component_scores.append((component, avg_impact, result.relative_importance))
        
        # Sort by average performance drop (higher = more important)
        component_scores.sort(key=lambda x: x[1], reverse=True)
        ablation_summary['component_importance_ranking'] = [
            {'component': comp, 'avg_impact': impact, 'importance': rel_imp}
            for comp, impact, rel_imp in component_scores
        ]
        
        # Identify critical components (>10% performance drop)
        critical_threshold = 0.1
        ablation_summary['critical_components'] = [
            comp for comp, impact, _ in component_scores if impact > critical_threshold
        ]
        
        # Performance impact analysis
        for component, result in ablation_data.items():
            ablation_summary['performance_impact_analysis'][component] = {
                'baseline_performance': result.baseline_performance,
                'ablated_performance': result.ablated_performance,
                'performance_drop': result.performance_drop,
                'critical_metrics': result.critical_metrics,
                'statistical_significance': {
                    metric: test.is_significant 
                    for metric, test in result.significance_tests.items()
                }
            }
        
        return ablation_summary
    
    def generate_publication_ready_plots(self, publication_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate publication-ready plots and return file paths"""
        
        plots = {}
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
            
            # Performance comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            experimental_results = publication_data['experimental_results']
            conditions = list(experimental_results.keys())
            metrics = ['preservation_effectiveness', 'processing_latency_ms', 'memory_overhead_mb', 'information_loss']
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx // 2, idx % 2]
                
                means = []
                stds = []
                labels = []
                
                for condition in conditions:
                    aggregated = experimental_results[condition]['aggregated_metrics']
                    if metric in aggregated:
                        means.append(aggregated[metric]['mean'])
                        stds.append(aggregated[metric]['std'])
                        labels.append(condition.replace('_', ' ').title())
                
                if means:
                    x_pos = np.arange(len(labels))
                    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                    ax.set_xlabel('Method')
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar, mean in zip(bars, means):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{mean:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            performance_plot_path = 'performance_comparison.png'
            plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['performance_comparison'] = performance_plot_path
            
            # Effect size visualization
            if 'effect_size_analysis' in publication_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                effect_sizes = publication_data['effect_size_analysis']['cohens_d']
                if effect_sizes:
                    metrics = []
                    values = []
                    
                    for key, value in effect_sizes.items():
                        metrics.append(key.split('_')[-1])  # Extract metric name
                        values.append(value)
                    
                    # Create horizontal bar plot
                    y_pos = np.arange(len(metrics))
                    bars = ax.barh(y_pos, values, alpha=0.7)
                    
                    # Color bars based on effect size magnitude
                    for bar, value in zip(bars, values):
                        if abs(value) < 0.2:
                            bar.set_color('lightgray')
                        elif abs(value) < 0.5:
                            bar.set_color('lightblue')
                        elif abs(value) < 0.8:
                            bar.set_color('orange')
                        else:
                            bar.set_color('red')
                    
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
                    ax.set_xlabel("Cohen's d (Effect Size)")
                    ax.set_title('Effect Size Analysis')
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='Small Effect')
                    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Medium Effect')
                    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Large Effect')
                    ax.legend()
                    
                    plt.tight_layout()
                    effect_size_plot_path = 'effect_size_analysis.png'
                    plt.savefig(effect_size_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots['effect_size_analysis'] = effect_size_plot_path
            
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn not available, skipping plot generation")
        except Exception as e:
            self.logger.error(f"Plot generation failed: {e}")
        
        return plots


class AblationStudyFramework:
    """Framework for conducting systematic ablation studies"""
    
    def __init__(self, preservation_guard, config: Dict[str, Any] = None):
        self.preservation_guard = preservation_guard
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + ".AblationStudyFramework")
    
    async def conduct_ablation_study(self, test_scenarios: List[Dict[str, Any]],
                                   components_to_ablate: List[str],
                                   num_repetitions: int = 20) -> Dict[str, AblationStudyResult]:
        """Conduct systematic ablation study"""
        
        ablation_results = {}
        
        # Get baseline performance (all components enabled)
        self.logger.info("Measuring baseline performance with all components")
        baseline_performance = await self._measure_baseline_performance(test_scenarios, num_repetitions)
        
        # Test each component ablation
        for component in components_to_ablate:
            self.logger.info(f"Testing ablation of component: {component}")
            
            ablated_performance = await self._measure_ablated_performance(
                test_scenarios, component, num_repetitions
            )
            
            # Calculate performance drops
            performance_drop = {}
            for metric in baseline_performance.keys():
                baseline_value = baseline_performance[metric]['mean']
                ablated_value = ablated_performance[metric]['mean']
                
                if baseline_value != 0:
                    drop = (baseline_value - ablated_value) / baseline_value
                else:
                    drop = 0
                
                performance_drop[metric] = drop
            
            # Calculate relative importance
            avg_drop = np.mean(list(performance_drop.values()))
            relative_importance = avg_drop
            
            # Identify critical metrics (>20% drop)
            critical_metrics = [
                metric for metric, drop in performance_drop.items() 
                if drop > 0.2
            ]
            
            # Statistical significance testing
            significance_tests = {}
            for metric in baseline_performance.keys():
                baseline_values = baseline_performance[metric]['raw_values']
                ablated_values = ablated_performance[metric]['raw_values']
                
                # Perform t-test
                stat, p_val = stats.ttest_ind(baseline_values, ablated_values)
                
                # Calculate effect size
                pooled_std = np.sqrt((np.var(baseline_values) + np.var(ablated_values)) / 2)
                cohens_d = (np.mean(baseline_values) - np.mean(ablated_values)) / pooled_std if pooled_std > 0 else 0
                
                significance_tests[metric] = StatisticalTest(
                    test_name="Independent t-test",
                    test_statistic=stat,
                    p_value=p_val,
                    effect_size=cohens_d,
                    confidence_interval=(0, 0),  # Simplified
                    sample_size=len(baseline_values) + len(ablated_values)
                )
            
            # Create ablation result
            ablation_results[component] = AblationStudyResult(
                component_name=component,
                baseline_performance={k: v['mean'] for k, v in baseline_performance.items()},
                ablated_performance={k: v['mean'] for k, v in ablated_performance.items()},
                performance_drop=performance_drop,
                relative_importance=relative_importance,
                critical_metrics=critical_metrics,
                significance_tests=significance_tests,
                confidence_level=0.95
            )
        
        return ablation_results
    
    async def _measure_baseline_performance(self, test_scenarios: List[Dict[str, Any]],
                                          num_repetitions: int) -> Dict[str, Dict[str, Any]]:
        """Measure baseline performance with all components enabled"""
        
        return await self._measure_performance_with_config(
            test_scenarios, {}, num_repetitions, "baseline"
        )
    
    async def _measure_ablated_performance(self, test_scenarios: List[Dict[str, Any]],
                                         ablated_component: str, num_repetitions: int) -> Dict[str, Dict[str, Any]]:
        """Measure performance with specific component ablated"""
        
        ablation_config = self._get_ablation_config(ablated_component)
        return await self._measure_performance_with_config(
            test_scenarios, ablation_config, num_repetitions, f"ablated_{ablated_component}"
        )
    
    def _get_ablation_config(self, component: str) -> Dict[str, Any]:
        """Get configuration for ablating specific component"""
        
        ablation_configs = {
            'neural_encoder': {'enable_neural_encoder': False},
            'adaptive_optimizer': {'enable_adaptive_optimizer': False},
            'info_analyzer': {'enable_info_analyzer': False},
            'graph_analyzer': {'enable_graph_analyzer': False},
            'semantic_preservation': {'enable_semantic_preservation': False},
            'privacy_utility_optimizer': {'enable_privacy_utility_optimizer': False}
        }
        
        return ablation_configs.get(component, {})
    
    async def _measure_performance_with_config(self, test_scenarios: List[Dict[str, Any]],
                                             config: Dict[str, Any], num_repetitions: int,
                                             config_name: str) -> Dict[str, Dict[str, Any]]:
        """Measure performance with specific configuration"""
        
        all_measurements = defaultdict(list)
        
        for scenario in test_scenarios:
            for rep in range(num_repetitions):
                try:
                    # Apply processing transformations
                    processed_scenario = await self._apply_processing_transformations(scenario)
                    
                    # Configure system
                    configured_guard = self._configure_guard_with_ablation(config)
                    
                    # Measure performance
                    assessment = await configured_guard.assess_preservation_impact(
                        scenario, processed_scenario, f"{config_name}_rep_{rep}"
                    )
                    
                    # Collect metrics
                    all_measurements['preservation_effectiveness'].append(assessment.preservation_effectiveness)
                    all_measurements['processing_time_ms'].append(assessment.processing_time_ms)
                    all_measurements['confidence_score'].append(assessment.confidence_score)
                    all_measurements['information_loss'].append(assessment.information_loss)
                    
                except Exception as e:
                    self.logger.error(f"Performance measurement failed: {e}")
        
        # Calculate aggregated statistics
        performance_stats = {}
        for metric, values in all_measurements.items():
            if values:
                performance_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'raw_values': values
                }
        
        return performance_stats
    
    def _configure_guard_with_ablation(self, ablation_config: Dict[str, Any]):
        """Configure preservation guard with ablation settings"""
        
        # Create a copy of the preservation guard with modified configuration
        # In practice, this would involve more sophisticated configuration management
        configured_guard = copy.deepcopy(self.preservation_guard)
        
        # Apply ablation configuration
        for setting, value in ablation_config.items():
            if hasattr(configured_guard, setting):
                setattr(configured_guard, setting, value)
        
        return configured_guard
    
    async def _apply_processing_transformations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply standard processing transformations"""
        
        processed = copy.deepcopy(data)
        
        # Remove debug fields
        processed.pop('debug_info', None)
        
        # Hash sensitive fields
        if 'user_id' in processed:
            processed['user_id'] = f"hash_{hash(processed['user_id'])}"
        
        return processed


# Export classes for integration
__all__ = [
    'ResearchMetricsCollector',
    'PublicationMetricsGenerator', 
    'AblationStudyFramework',
    'StatisticalTest',
    'ExperimentalResult',
    'ComparativeAnalysis',
    'AblationStudyResult',
    'ResearchMetricType',
    'ExperimentalCondition'
]